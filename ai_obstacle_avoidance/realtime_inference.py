import time

import cv2
import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# ==========================================
# 設定
# ==========================================

MODEL_PATH = "models/obstacle_cnn_best.pth"

CLASSES = ["none", "left", "center", "right"]

WIDTH = 160
HEIGHT = 120

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 学習時と同じCNN
# ==========================================

class ObstacleCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(CLASSES))
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# ==========================================
# ROS 2ノード
# ==========================================

class RealtimeInference(Node):

    def __init__(self):

        super().__init__("realtime_inference")

        self.bridge = CvBridge()

        self.model = ObstacleCNN().to(device)

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )

        if checkpoint["classes"] != CLASSES:
            raise RuntimeError("Class order mismatch")

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.subscription = self.create_subscription(
            Image,
            "/camera/image",
            self.image_callback,
            qos_profile_sensor_data
        )

        self.last_log_time = 0.0

        self.get_logger().info(
            f"CNN loaded. Device: {device}"
        )

        cv2.namedWindow(
            "CNN Obstacle Recognition",
            cv2.WINDOW_NORMAL
        )

        cv2.waitKey(1)

    def preprocess(self, image):

        # 640x480 → 160x120
        resized = cv2.resize(
            image,
            (WIDTH, HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        # BGR → RGB
        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        # HWC → CHW
        array = np.ascontiguousarray(
            rgb.transpose(2, 0, 1)
        )

        tensor = (
            torch.from_numpy(array)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )

        return tensor

    def image_callback(self, msg):

        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8"
        )

        input_tensor = self.preprocess(image)

        # CNNによる推論
        with torch.inference_mode():

            outputs = self.model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        label = CLASSES[prediction.item()]
        score = confidence.item()

        # 表示用画像を作成
        display = image.copy()

        text = f"Prediction: {label} ({score:.1%})"

        cv2.putText(
            display,
            text,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        # 3分割の境界線
        height, width = display.shape[:2]

        cv2.line(
            display,
            (width // 3, 0),
            (width // 3, height),
            (255, 255, 0),
            1
        )

        cv2.line(
            display,
            (width * 2 // 3, 0),
            (width * 2 // 3, height),
            (255, 255, 0),
            1
        )

        cv2.imshow("CNN Obstacle Recognition", display)
        cv2.waitKey(1)

        # 1秒ごとに推論結果をログ出力
        now = time.monotonic()

        if now - self.last_log_time >= 1.0:

            self.get_logger().info(
                f"Prediction: {label}, "
                f"confidence: {score:.3f}"
            )

            self.last_log_time = now


def main():

    rclpy.init()

    node = None

    try:

        node = RealtimeInference()

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        if node is not None:
            node.destroy_node()

        cv2.destroyAllWindows()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
