import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraTest(Node):

    def __init__(self):
        super().__init__('camera_test')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info('Waiting for camera images...')

    def image_callback(self, msg):

        # ROS 2 Image → OpenCV画像
        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        # カメラ画像を表示
        cv2.imshow('Robot Camera', image)
        cv2.waitKey(1)


def main():

    rclpy.init()

    node = CameraTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
