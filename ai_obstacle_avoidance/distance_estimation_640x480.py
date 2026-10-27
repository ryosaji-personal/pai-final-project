import math
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# ==========================================
# カメラ・障害物の設定
# ==========================================

IMAGE_WIDTH = 640

# world.sdfのhorizontal_fov
HORIZONTAL_FOV = 1.047

# 赤い箱の実際の幅 [m]
BOX_WIDTH = 0.4

# 焦点距離 [pixel]
FX = IMAGE_WIDTH / (
    2.0 * math.tan(HORIZONTAL_FOV / 2.0)
)

# 小さなノイズを除外するための面積しきい値
MIN_AREA = 150

WINDOW_NAME = "Obstacle Distance Estimation"


# ==========================================
# 赤い箱の検出
# ==========================================

def detect_red_box(image):

    # BGR → HSV
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # 赤色はHSV色相環の両端に存在する
    lower_red1 = np.array([0, 100, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 70])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(
        hsv, lower_red1, upper_red1
    )

    mask2 = cv2.inRange(
        hsv, lower_red2, upper_red2
    )

    mask = cv2.bitwise_or(mask1, mask2)

    # 小さなノイズを除去
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # 赤色領域の輪郭を検出
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, mask

    # 最大の赤色領域を選択
    contour = max(
        contours,
        key=cv2.contourArea
    )

    if cv2.contourArea(contour) < MIN_AREA:
        return None, mask

    x, y, w, h = cv2.boundingRect(contour)

    return (x, y, w, h), mask


# ==========================================
# ROS 2ノード
# ==========================================

class DistanceEstimator(Node):

    def __init__(self):

        super().__init__(
            "distance_estimator"
        )

        self.bridge = CvBridge()

        self.create_subscription(
            Image,
            "/camera/image",
            self.image_callback,
            qos_profile_sensor_data
        )

        self.last_log_time = 0.0

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL
        )

        self.get_logger().info(
            f"Distance estimator started. FX={FX:.1f}"
        )

    def image_callback(self, msg):

        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8"
        )

        display = image.copy()

        box, mask = detect_red_box(image)

        if box is not None:

            x, y, w, h = box

            # カメラ前方方向の概算距離
            distance = FX * BOX_WIDTH / w

            # 障害物の画像上の中心位置
            center_x = x + w / 2.0

            # 画像中心からの水平方向の角度
            angle = math.degrees(
                math.atan(
                    (center_x - image.shape[1] / 2.0)
                    / FX
                )
            )

            # 左右端で切れている場合は幅を測れない
            clipped = (
                x <= 1
                or x + w >= image.shape[1] - 1
            )

            # 障害物の検出枠を描画
            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            if clipped:
                distance_text = "Distance: unavailable (clipped)"
            else:
                distance_text = (
                    f"Approx. distance: {distance:.2f} m"
                )

            angle_text = (
                f"Direction: {angle:+.1f} deg"
            )

        else:

            distance_text = "Obstacle: Not detected"
            angle_text = ""

        # 画像上に情報を表示
        cv2.putText(
            display,
            distance_text,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        if angle_text:

            cv2.putText(
                display,
                angle_text,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        cv2.imshow(
            WINDOW_NAME,
            display
        )

        cv2.waitKey(1)

        # 1秒ごとにログ出力
        now = time.monotonic()

        if now - self.last_log_time >= 1.0:

            self.get_logger().info(
                distance_text
            )

            self.last_log_time = now


def main():

    rclpy.init()

    node = None

    try:

        node = DistanceEstimator()

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
