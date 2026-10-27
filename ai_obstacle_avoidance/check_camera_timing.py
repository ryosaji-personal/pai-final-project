import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class CameraTiming(Node):
    def __init__(self):
        super().__init__("camera_timing")

        self.previous_wall = None
        self.previous_stamp = None
        self.count = 0

        self.create_subscription(
            Image,
            "/camera/image",
            self.callback,
            qos_profile_sensor_data
        )

    def callback(self, msg):
        wall = time.monotonic()

        stamp = (
            msg.header.stamp.sec
            + msg.header.stamp.nanosec * 1e-9
        )

        if self.previous_wall is not None:
            wall_dt = wall - self.previous_wall
            stamp_dt = stamp - self.previous_stamp

            if wall_dt > 0.3 or stamp_dt > 0.3:
                print(
                    f"GAP: receive={wall_dt:.3f}s, "
                    f"stamp={stamp_dt:.3f}s",
                    flush=True
                )

        self.previous_wall = wall
        self.previous_stamp = stamp

        self.count += 1

        if self.count % 50 == 0:
            print(
                f"Received {self.count} frames",
                flush=True
            )


def main():
    rclpy.init()
    node = CameraTiming()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
