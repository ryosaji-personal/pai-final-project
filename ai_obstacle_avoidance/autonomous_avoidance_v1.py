import time

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from realtime_inference import ObstacleCNN
from distance_estimation import (
    detect_red_box,
    FX,
    BOX_WIDTH,
)


# ==========================================
# 設定
# ==========================================

MODEL_PATH = "models/obstacle_cnn_best.pth"

CLASSES = ["none", "left", "center", "right"]

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120

# 最初は制御指令を表示するだけ
ENABLE_MOTION = True

# 制御パラメータ
FORWARD_SPEED = 0.05
TURN_SPEED = 0.25

AVOID_DISTANCE = 0.65
CONFIDENCE_THRESHOLD = 0.70

# 実走行時の最大試験時間
MAX_RUN_SECONDS = 12.0

WINDOW_NAME = "Autonomous Obstacle Avoidance"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class AutonomousAvoidance(Node):

    def __init__(self):

        super().__init__("autonomous_avoidance")

        self.bridge = CvBridge()

        self.model = ObstacleCNN().to(DEVICE)

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )

        if checkpoint["classes"] != CLASSES:
            raise RuntimeError("Class order mismatch")

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.cmd_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )

        self.last_image = None
        self.last_image_time = None
        self.last_log_time = 0.0

        self.finished = False
        self.start_time = None

        self.create_subscription(
            Image,
            "/camera/image",
            self.image_callback,
            qos_profile_sensor_data
        )

        self.timer = self.create_timer(
            0.1,
            self.control_callback
        )

        # cv2.namedWindow(
        #     WINDOW_NAME,
        #     cv2.WINDOW_NORMAL
        # )

        self.get_logger().info(
            f"Model loaded. Device: {DEVICE}"
        )

        self.get_logger().info(
            f"ENABLE_MOTION = {ENABLE_MOTION}"
        )

    def image_callback(self, msg):

        self.last_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8"
        )

        self.last_image_time = time.monotonic()

    def predict(self, image):

        resized = cv2.resize(
            image,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        array = np.ascontiguousarray(
            rgb.transpose(2, 0, 1)
        )

        tensor = (
            torch.from_numpy(array)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.inference_mode():

            output = self.model(tensor)

            probabilities = torch.softmax(
                output, dim=1
            )

            confidence, prediction = torch.max(
                probabilities, dim=1
            )

        return (
            CLASSES[prediction.item()],
            confidence.item()
        )

    def publish_velocity(self, linear=0.0, angular=0.0):

        cmd = Twist()

        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)

        self.cmd_pub.publish(cmd)

    def decide_action(self, label, confidence, box):

        # CNNの確信度が低い場合は停止
        if confidence < CONFIDENCE_THRESHOLD:
            return "STOP_LOW_CONF", 0.0, 0.0, None

        # 赤い箱が検出されている場合
        if box is not None:

            x, y, w, h = box

            # 画像の左右端に接している場合、
            # 箱の幅を正しく測定できない
            clipped = (
                x <= 1
                or x + w >= 639
            )

            if clipped:
                return "STOP_CLIPPED", 0.0, 0.0, None

            distance = FX * BOX_WIDTH / w

            if distance <= AVOID_DISTANCE:

                if label == "left":
                    return "TURN_RIGHT", 0.0, -TURN_SPEED, distance

                elif label == "right":
                    return "TURN_LEFT", 0.0, TURN_SPEED, distance

                elif label == "center":
                    return "TURN_RIGHT", 0.0, -TURN_SPEED, distance

                else:
                    # 赤い領域は検出したが、CNNはnoneと判断した
                    return "STOP_CONFLICT", 0.0, 0.0, distance

            # 遠い場合は低速前進
            return "FORWARD", FORWARD_SPEED, 0.0, distance

        # 赤い領域はないがCNNが障害物ありと判断
        if label != "none":
            return "STOP_NO_DISTANCE", 0.0, 0.0, None

        # どちらも障害物なし
        return "FORWARD", FORWARD_SPEED, 0.0, None

    def control_callback(self):

        if self.finished:
            return

        now = time.monotonic()

        # カメラ画像が届かない場合
        if self.last_image_time is None:
            return

        if now - self.last_image_time > 1.0:

            if ENABLE_MOTION:
                self.publish_velocity()

            self.get_logger().warning(
                "Camera timeout. STOP"
            )
            return

        # 画像取得後に時間制限を開始
        if self.start_time is None:
            self.start_time = now

        if (
            ENABLE_MOTION
            and now - self.start_time >= MAX_RUN_SECONDS
        ):

            self.publish_velocity()
            self.finished = True

            self.get_logger().info(
                "Maximum run time reached. STOP"
            )
            return

        image = self.last_image.copy()

        # CNNによる分類
        label, confidence = self.predict(image)

        # OpenCVによる赤い箱の検出
        box, mask = detect_red_box(image)

        # 制御判断
        action, linear, angular, distance = (
            self.decide_action(label, confidence, box)
        )

        # 自律走行を有効にした場合だけ送信
        if ENABLE_MOTION:
            self.publish_velocity(linear, angular)

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

        lines = [
            f"CNN: {label} ({confidence:.1%})",
            f"Distance: {distance_text}",
            f"Action: {action}",
            f"Motion: {'ON' if ENABLE_MOTION else 'OFF'}",
        ]

        for i, text in enumerate(lines):

            cv2.putText(
                display,
                text,
                (10, 30 + i * 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        cv2.imshow(
            WINDOW_NAME,
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            self.finished = True

        # ログは1秒ごと
        if now - self.last_log_time >= 1.0:

            self.get_logger().info(
                f"CNN={label}, "
                f"confidence={confidence:.2f}, "
                f"distance={distance_text}, "
                f"action={action}"
            )

            self.last_log_time = now

    def close(self):

        # 停止指令を複数回送る
        if ENABLE_MOTION:
            for _ in range(5):
                self.publish_velocity()
                time.sleep(0.05)


def main():

    rclpy.init()

    node = None

    try:

        node = AutonomousAvoidance()

        while rclpy.ok() and not node.finished:
            rclpy.spin_once(
                node,
                timeout_sec=0.1
            )

    except KeyboardInterrupt:
        pass

    finally:

        if node is not None:
            node.close()
            node.destroy_node()

        cv2.destroyAllWindows()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
