import os
import time
import cv2
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageCollector(Node):

    def __init__(self):
        super().__init__('image_collector')

        self.bridge = CvBridge()

        self.labels = {
            ord('0'): 'none',
            ord('1'): 'left',
            ord('2'): 'center',
            ord('3'): 'right',
        }

        self.subscription = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info('Image collector started')

    def image_callback(self, msg):

        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        display_image = image.copy()

        cv2.putText(
            display_image,
            '0:None  1:Left  2:Center  3:Right  Q:Quit',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.imshow('Training Data Collector', display_image)

        key = cv2.waitKey(1) & 0xFF

        if key in self.labels:

            label = self.labels[key]

            directory = os.path.join('dataset', label)
            os.makedirs(directory, exist_ok=True)

            filename = os.path.join(
                directory,
                f'{time.time_ns()}.png'
            )

            cv2.imwrite(filename, image)

            self.get_logger().info(
                f'Saved: {filename}'
            )

        elif key == ord('q'):
            raise KeyboardInterrupt


def main():

    rclpy.init()
    node = ImageCollector()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        cv2.destroyAllWindows()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
