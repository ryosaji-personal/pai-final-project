import csv
import math
import time
from pathlib import Path

import cv2
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# ==============================================
# 撮影設定
# ==============================================

OUTPUT_DIR = Path("dataset/raw")

# 最初は200枚のテスト
# テスト成功後、[0.0, 0.15, 0.30, 0.45, 0.60] に変更
POSITIONS = [0.0, 0.15, 0.30, 0.45, 0.60]

IMAGES_PER_POSITION = 200

SCAN_START_DEG = -75.0
SCAN_END_DEG = 75.0

ANGULAR_SPEED = 0.08
LINEAR_SPEED = 0.08


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def get_yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


class AutoCollector(Node):

    def __init__(self):
        super().__init__("auto_collector")

        self.bridge = CvBridge()

        self.image = None
        self.image_stamp = None
        self.last_saved_stamp = None

        self.odom = None
        self.last_image_time = None
        self.last_odom_time = None

        self.origin = None
        self.initial_yaw = None

        self.position_index = 0
        self.image_index = 0
        self.total_saved = 0

        self.state = "waiting"
        self.finished = False

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # 既存の撮影画像との混在を防止
        if any(OUTPUT_DIR.iterdir()):
            raise RuntimeError(
                f"{OUTPUT_DIR} is not empty. "
                "Use an empty directory for this run."
            )

        self.csv_file = open(
            OUTPUT_DIR / "metadata.csv",
            "w",
            newline=""
        )

        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            "filename",
            "position_index",
            "image_index",
            "x",
            "y",
            "yaw_deg",
            "image_sec",
            "image_nanosec"
        ])

        self.cmd_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )

        self.create_subscription(
            Image,
            "/camera/image",
            self.image_callback,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Odometry,
            "/model/demoRobot/odometry",
            self.odom_callback,
            qos_profile_sensor_data
        )

        self.timer = self.create_timer(
            0.05,
            self.control_callback
        )

        self.get_logger().info(
            "AutoCollector started"
        )

    def image_callback(self, msg):

        self.last_image_time = time.monotonic()

        self.image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8"
        )

        self.image_stamp = (
            msg.header.stamp.sec,
            msg.header.stamp.nanosec
        )

    def odom_callback(self, msg):

        self.odom = msg
        self.last_odom_time = time.monotonic()

    def publish_velocity(self, linear=0.0, angular=0.0):

        cmd = Twist()

        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)

        self.cmd_pub.publish(cmd)

    def current_pose(self):

        position = self.odom.pose.pose.position
        orientation = self.odom.pose.pose.orientation

        return (
            position.x,
            position.y,
            get_yaw(orientation)
        )

    def relative_yaw(self, yaw):

        return normalize_angle(
            yaw - self.initial_yaw
        )

    def move_to_angle(self, target, speed=0.25):

        _, _, yaw = self.current_pose()

        current = self.relative_yaw(yaw)

        error = normalize_angle(target - current)

        if abs(error) < 0.012:
            self.publish_velocity()
            return True

        angular = max(
            -speed,
            min(speed, 2.0 * error)
        )

        self.publish_velocity(angular=angular)

        return False

    def save_image(self):

        x, y, yaw = self.current_pose()

        filename = (
            f"pos{self.position_index:02d}_"
            f"img{self.image_index:03d}.png"
        )

        path = OUTPUT_DIR / filename

        if not cv2.imwrite(str(path), self.image):
            raise RuntimeError(
                f"Failed to save {path}"
            )

        self.csv_writer.writerow([
            filename,
            self.position_index,
            self.image_index,
            x,
            y,
            math.degrees(yaw),
            self.image_stamp[0],
            self.image_stamp[1]
        ])

        self.csv_file.flush()

        self.last_saved_stamp = self.image_stamp

        self.image_index += 1
        self.total_saved += 1

        if self.image_index % 20 == 0:
            self.get_logger().info(
                f"Position {self.position_index}: "
                f"{self.image_index}/{IMAGES_PER_POSITION}, "
                f"Total: {self.total_saved}"
            )

    def control_callback(self):

        if self.finished:
            return

        if self.odom is None or self.image is None:
            return

        now = time.monotonic()

        # センサー情報が途絶えた場合は停止
        if (
            now - self.last_image_time > 3.0
            or now - self.last_odom_time > 3.0
        ):
            self.get_logger().error(
                "Camera or odometry data timeout"
            )
            self.finish()
            return

        x, y, yaw = self.current_pose()

        # 初期位置と初期姿勢を取得
        if self.state == "waiting":

            self.origin = (x, y)
            self.initial_yaw = yaw

            self.get_logger().info(
                f"Origin: x={x:.3f}, "
                f"y={y:.3f}, "
                f"yaw={math.degrees(yaw):.1f}"
            )

            self.state = "scan_start"

        # -75度へ移動
        elif self.state == "scan_start":

            target = math.radians(SCAN_START_DEG)

            if self.move_to_angle(target):

                self.image_index = 0

                self.get_logger().info(
                    f"Start scanning position "
                    f"{self.position_index}"
                )

                self.state = "scanning"

        # -75度から+75度へ旋回しながら撮影
        elif self.state == "scanning":

            current = self.relative_yaw(yaw)

            start = math.radians(SCAN_START_DEG)
            end = math.radians(SCAN_END_DEG)

            ratio = (
                self.image_index
                / (IMAGES_PER_POSITION - 1)
            )

            target_angle = (
                start + (end - start) * ratio
            )

            # 新しい画像かつ撮影角度に到達
            if (
                self.image_stamp != self.last_saved_stamp
                and current >= target_angle - 0.015
            ):
                self.save_image()

            if self.image_index >= IMAGES_PER_POSITION:

                self.publish_velocity()

                self.get_logger().info(
                    f"Position {self.position_index} completed"
                )

                self.state = "face_forward"

            else:

                # 画像が欠落した場合に旋回しすぎないよう制限
                if current > end + 0.035:
                    self.get_logger().error(
                        "Scan ended before all images were saved"
                    )
                    self.finish()
                    return

                self.publish_velocity(
                    angular=ANGULAR_SPEED
                )

        # 正面方向へ戻る
        elif self.state == "face_forward":

            if self.move_to_angle(0.0):

                self.position_index += 1

                if self.position_index >= len(POSITIONS):

                    self.finish()

                else:

                    self.state = "moving"

        # 次の撮影位置まで前進
        elif self.state == "moving":

            dx = x - self.origin[0]
            dy = y - self.origin[1]

            # 初期姿勢方向への移動距離
            distance = (
                dx * math.cos(self.initial_yaw)
                + dy * math.sin(self.initial_yaw)
            )

            target = POSITIONS[self.position_index]

            error = target - distance

            if error <= 0.005:

                self.publish_velocity()

                self.get_logger().info(
                    f"Arrived at position "
                    f"{self.position_index}"
                )

                self.state = "scan_start"

            else:

                heading_error = normalize_angle(
                    self.initial_yaw - yaw
                )

                linear = min(
                    LINEAR_SPEED,
                    max(0.015, 0.8 * error)
                )

                angular = max(
                    -0.2,
                    min(0.2, 2.0 * heading_error)
                )

                self.publish_velocity(
                    linear=linear,
                    angular=angular
                )

    def finish(self):

        self.publish_velocity()

        self.finished = True

        self.get_logger().info(
            f"Finished. Saved {self.total_saved} images."
        )

    def close(self):

        self.publish_velocity()
        self.csv_file.close()


def main():

    rclpy.init()

    node = None

    try:

        node = AutoCollector()

        while rclpy.ok() and not node.finished:
            rclpy.spin_once(
                node,
                timeout_sec=0.1
            )

    except KeyboardInterrupt:
        print("Interrupted")

    finally:

        if node is not None:
            node.close()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
