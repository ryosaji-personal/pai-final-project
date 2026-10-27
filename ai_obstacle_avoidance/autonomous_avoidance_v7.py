import math
import time

import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from std_msgs.msg import String

from autonomous_avoidance_v1 import (
    AutonomousAvoidance,
    DEVICE,
)

# ==========================================
# 制御パラメータ
# ==========================================

ENABLE_GUI = False

FORWARD_SPEED = 0.05
TURN_SPEED = 0.25

AVOID_DISTANCE = 0.65
CONFIDENCE_THRESHOLD = 0.70

# 右旋回45度
TURN_ANGLE_DEG = 45.0

# 旋回角度の許容誤差
ANGLE_TOLERANCE_DEG = 1.5

# 実験の時間制限（秒）
MAX_RUN_SECONDS = 90.0

# 斜め移動の目標距離 [m]
MOVE_ASIDE_DISTANCE = 0.70

# 斜め移動の速度 [m/s]
MOVE_ASIDE_SPEED = 0.05

# 障害物の横を通過する距離 [m]
PASS_DISTANCE = 1.30

# 通過時の速度 [m/s]
PASS_SPEED = 0.05

# センサーのタイムアウト
SENSOR_TIMEOUT = 1.0

# 遠距離ではCNNの低信頼度を許容して接近する
FAR_DISTANCE = 1.0

WINDOW_NAME = "Autonomous Avoidance V2"

ENABLE_MOTION_V6 = True

# ==========================================
# 角度計算
# ==========================================

def normalize_angle(angle):
    return math.atan2(
        math.sin(angle),
        math.cos(angle)
    )


def get_yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


# ==========================================
# ROS 2ノード
# ==========================================

class AutonomousAvoidanceV2(AutonomousAvoidance):

    def __init__(self):

        super().__init__()

        # CNNの認識結果を重ねた画像を配信
        self.debug_pub = self.create_publisher(
            Image,
            "/debug/image",
            10
        )

        self.state = "APPROACH"

        self.odom = None
        self.last_odom_time = None

        self.turn_start_yaw = None
        self.target_yaw = None

        # 回避開始前の進行方向
        self.original_heading = None

        # MOVE_ASIDE開始時の位置
        self.move_start_x = None
        self.move_start_y = None

        # MOVE_ASIDE開始時の方向
        self.move_heading = None

        self.run_start_time = None

        self.pass_start_x = None
        self.pass_start_y = None
        self.pass_heading = None

        self.create_subscription(
            Odometry,
            "/model/demoRobot/odometry",
            self.odom_callback,
            qos_profile_sensor_data
        )

        # 親クラスの制御タイマーは実際には使わず、
        # このクラスの状態遷移制御で動作させる
        self.timer.cancel()

        self.control_timer = self.create_timer(
            0.1,
            self.control_v2
        )

        self.get_logger().info(
            "State machine initialized: APPROACH"
        )

        self.status_pub = self.create_publisher(
            String,
            "/debug/status",
            10
        )

    def odom_callback(self, msg):

        self.odom = msg
        self.last_odom_time = time.monotonic()

    def current_yaw(self):

        return get_yaw(
            self.odom.pose.pose.orientation
        )

    def change_state(self, new_state):

        if self.state == new_state:
            return

        self.publish_velocity()

        self.get_logger().info(
            f"STATE: {self.state} -> {new_state}"
        )

        self.state = new_state

    def control_v2(self):

        if self.finished:
            return

        now = time.monotonic()

        # センサーの初回受信を待つ
        if (
            self.last_image_time is None
            or self.odom is None
        ):
            return

        # センサーの更新が止まった場合
        image_age = now - self.last_image_time
        odom_age = now - self.last_odom_time

        if (
            image_age > SENSOR_TIMEOUT
            or odom_age > SENSOR_TIMEOUT
        ):
            
            self.publish_velocity()

            self.get_logger().error(
                f"Sensor timeout: "
                f"image_age={image_age:.3f}s, "
                f"odom_age={odom_age:.3f}s, "
                f"threshold={SENSOR_TIMEOUT:.3f}s"
            )

            self.change_state("STOP")
            self.finished = True
            return

        if self.run_start_time is None:
            self.run_start_time = now

        # 実験時間の上限
        if now - self.run_start_time > MAX_RUN_SECONDS:

            self.change_state("STOP")

            self.get_logger().warning(
                "Maximum run time reached"
            )

            self.finished = True
            return

        image = self.last_image.copy()

        # CNNによる障害物の分類
        label, confidence = self.predict(image)

        # 距離推定
        from distance_estimation import (
            detect_red_box,
            FX,
            BOX_WIDTH
        )

        box, _ = detect_red_box(image)

        distance = None
        clipped = False

        if box is not None:

            x, y, w, h = box

            clipped = (
                x <= 1
                or x + w >= image.shape[1] - 1
            )

            if not clipped and w > 0:
                distance = FX * BOX_WIDTH / w

        action = "STOP"

        # ======================================
        # APPROACH：障害物へ接近
        # ======================================

        if self.state == "APPROACH":

            if (
                confidence < CONFIDENCE_THRESHOLD
                and distance is not None
                and distance > FAR_DISTANCE
                and not clipped
            ):
                # 遠距離ではCNNの低信頼度を許容し、低速で接近する
                self.publish_velocity(
                    linear=FORWARD_SPEED * 0.5
                )
                action = "FORWARD_LOW_CONF"

            elif confidence < CONFIDENCE_THRESHOLD:

                self.publish_velocity()
                action = "WAIT_LOW_CONF"

            elif clipped:

                self.publish_velocity()
                action = "WAIT_CLIPPED"

            elif box is None and label != "none":

                self.publish_velocity()
                action = "WAIT_NO_DISTANCE"

            elif box is not None and label == "none":

                self.publish_velocity()
                action = "WAIT_CONFLICT"

            elif (
                label == "center"
                and distance is not None
                and distance <= AVOID_DISTANCE
            ):

                self.publish_velocity()

                self.turn_start_yaw = self.current_yaw()

                # 元の進行方向を記録
                self.original_heading = self.turn_start_yaw

                self.target_yaw = normalize_angle(
                    self.turn_start_yaw
                    - math.radians(TURN_ANGLE_DEG)
                )

                self.change_state("TURN_AWAY")

                action = "START_TURN"

            else:

                self.publish_velocity(
                    linear=FORWARD_SPEED
                )

                action = "FORWARD"

        # ======================================
        # TURN_AWAY：右旋回45度
        # ======================================

        elif self.state == "TURN_AWAY":

            yaw = self.current_yaw()

            error = normalize_angle(
                self.target_yaw - yaw
            )

            tolerance = math.radians(
                ANGLE_TOLERANCE_DEG
            )

            if abs(error) <= tolerance:

                # 45°旋回完了時の位置を記録
                position = self.odom.pose.pose.position

                self.move_start_x = position.x
                self.move_start_y = position.y

                # 斜め前進する方向を記録
                self.move_heading = yaw

                # MOVE_ASIDEへ遷移
                self.change_state("MOVE_ASIDE")

                action = "TURN_COMPLETE"

            else:

                # 目標角度に近づくほど減速する
                angular = max(
                    -TURN_SPEED,
                    min(
                        TURN_SPEED,
                        1.5 * error
                    )
                )

                self.publish_velocity(
                    angular=angular
                )

                action = "TURNING"

        # ======================================
        # MOVE_ASIDE：右斜め前方へ移動
        # ======================================

        elif self.state == "MOVE_ASIDE":

            position = self.odom.pose.pose.position

            # 旋回完了位置からの移動量
            dx = position.x - self.move_start_x
            dy = position.y - self.move_start_y

            # 移動開始時の進行方向への移動距離
            traveled = (
                dx * math.cos(self.move_heading)
                + dy * math.sin(self.move_heading)
            )

            remaining = MOVE_ASIDE_DISTANCE - traveled

            if remaining <= 0.01:

                self.get_logger().info(
                    f"MOVE_ASIDE completed: {traveled:.3f} m"
                )

                # 元の進行方向を目標角度に設定
                self.target_yaw = self.original_heading

                # 左旋回による方向復帰へ移行
                self.change_state("RETURN_HEADING")

                action = "MOVE_COMPLETE"

            else:

                # 残り距離に応じて減速
                linear = min(
                    MOVE_ASIDE_SPEED,
                    max(0.01, 0.5 * remaining)
                )

                # 45°旋回後の方向を維持
                heading_error = normalize_angle(
                    self.move_heading - self.current_yaw()
                )

                angular = max(
                    -0.15,
                    min(0.15, 2.0 * heading_error)
                )

                self.publish_velocity(
                    linear=linear,
                    angular=angular
                )

                action = f"MOVING {traveled:.2f}/{MOVE_ASIDE_DISTANCE:.2f}m"

        # ======================================
        # RETURN_HEADING：元の進行方向へ戻る
        # ======================================

        elif self.state == "RETURN_HEADING":

            yaw = self.current_yaw()

            # 元の進行方向までの角度差
            error = normalize_angle(
                self.target_yaw - yaw
            )

            tolerance = math.radians(
                ANGLE_TOLERANCE_DEG
            )

            if abs(error) <= tolerance:

                self.get_logger().info(
                    "RETURN_HEADING completed"
                )

                position = self.odom.pose.pose.position

                self.pass_start_x = position.x
                self.pass_start_y = position.y
                self.pass_heading = self.original_heading
                
                self.change_state("PASS_OBSTACLE")

                action = "HEADING_RESTORED"

            else:

                # 目標角度に近づくほど減速
                angular = max(
                    -TURN_SPEED,
                    min(
                    TURN_SPEED,
                    1.5 * error
                    )
                )
                
                # その場で旋回する
                self.publish_velocity(
                    linear=0.0,
                    angular=angular
                )

                action = "RETURNING_HEADING"

        # ======================================
        # PASS_OBSTACLE：障害物の横を通過
        # ======================================

        elif self.state == "PASS_OBSTACLE":

            position = self.odom.pose.pose.position

            dx = position.x - self.pass_start_x
            dy = position.y - self.pass_start_y

            traveled = (
                dx * math.cos(self.pass_heading)
                + dy * math.sin(self.pass_heading)
            )

            remaining = PASS_DISTANCE - traveled

            if remaining <= 0.01:

                self.get_logger().info(
                    f"PASS_OBSTACLE completed: {traveled:.3f} m"
                )

                self.change_state("STOP")

                action = "PASS_COMPLETE"
                self.finished = True

            else:

                linear = min(
                    PASS_SPEED,
                    max(0.01, 0.5 * remaining)
                )

                heading_error = normalize_angle(
                    self.pass_heading - self.current_yaw()
                )

                angular = max(
                    -0.15,
                    min(0.15, 2.0 * heading_error)
                )

                self.publish_velocity(
                    linear=linear,
                    angular=angular
                )

                action = (
                    f"PASSING {traveled:.2f}/{PASS_DISTANCE:.2f}m"
                )

        # ======================================
        # STOP
        # ======================================

        elif self.state == "STOP":

            self.publish_velocity()
            action = "STOP"
            self.finished = True

        # ======================================
        # 結果表示
        # ======================================

        display = image.copy()

        if box is not None:

            x, y, w, h = box

            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        distance_text = (
            f"{distance:.2f} m"
            if distance is not None
            else "N/A"
        )

        yaw_deg = math.degrees(
            self.current_yaw()
        )

        lines = [
            f"CNN: {label} ({confidence:.0%})",
            f"State: {self.state}",
            f"Dist: {distance_text}",
        ]

        for i, text in enumerate(lines):

            cv2.putText(
                display,
                text,
                (4, 14 + i * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

        # ROS 2 Imageメッセージへ変換
        debug_msg = self.bridge.cv2_to_imgmsg(
            display,
            encoding="bgr8"
        )

        # CNN認識結果付き画像を配信
        self.debug_pub.publish(debug_msg)

        status_msg = String()

        status_msg.data = (
            f"CNN: {label} ({confidence:.0%})\n"
            f"Distance: {distance_text}\n"
            f"State: {self.state}\n"
            f"Action: {action}"
        )

        self.status_pub.publish(status_msg)

        if ENABLE_GUI:

            cv2.imshow(
                WINDOW_NAME,
                display
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                self.change_state("STOP")
                self.finished = True

        # ログは1秒ごと
        if now - self.last_log_time >= 1.0:

            self.get_logger().info(
                f"state={self.state}, "
                f"CNN={label}, "
                f"distance={distance_text}, "
                f"yaw={yaw_deg:.1f}, "
                f"action={action}"
            )

            self.last_log_time = now

    def publish_velocity(self, linear=0.0, angular=0.0):

        if not ENABLE_MOTION_V6:
            return

        super().publish_velocity(
            linear=linear,
            angular=angular
        )


# ==========================================
# main
# ==========================================

def main():

    rclpy.init()

    node = None

    try:

        node = AutonomousAvoidanceV2()

        while rclpy.ok() and not node.finished:

            rclpy.spin_once(
                node,
                timeout_sec=0.1
            )

    except KeyboardInterrupt:
        pass

    finally:

        if node is not None:

            # 停止指令を送信
            for _ in range(5):
                node.publish_velocity()
                time.sleep(0.05)

            node.destroy_node()

        cv2.destroyAllWindows()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
