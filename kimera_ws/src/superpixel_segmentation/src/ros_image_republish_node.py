#!/usr/bin/env python
import argparse
import rospy
import message_filters
from sensor_msgs.msg import Image


class ImageRepublishNode(object):
    def __init__(self) -> None:
        super().__init__()
        parser = argparse.ArgumentParser()
        args = self.parse_arguments(parser)

        self.image_topic = args.image_topic
        self.depth_topic = args.depth_topic
        
        image_sub = message_filters.Subscriber(self.image_topic, Image, queue_size=1)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image, queue_size=1)
        
        self.image_pub = rospy.Publisher(self.image_topic + "_repub", Image, queue_size=1)
        self.depth_pub = rospy.Publisher(self.depth_topic + "_repub", Image, queue_size=1)
        
        ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, depth_sub],
            1,
            slop=1.0,
        )
        ts.registerCallback(self.img_callback)

    def parse_arguments(self, parser):
        # Parse arguments for global params
        parser.add_argument(
            "--image_topic",
            type=str,
            default="/timon_jetson_zed_node/left/image_rect_color",
            help="Where the streaming RGB images are received over ROS",
        )
        parser.add_argument(
            "--depth_topic",
            type=str,
            default="/timon_jetson_zed_node/depth/depth_registered",
            help="The topic where the robot depth images are published",
        )
        args = parser.parse_args()

        return args

    def img_callback(self, image_msg, depth_msg):
        print(f"Headers: {image_msg.header.stamp} {depth_msg.header.stamp}")

        self.image_pub(image_msg)
        self.depth_pub(depth_msg)

if __name__ == '__main__':

    rospy.init_node("image_republish_node")
    node_name = rospy.get_name()

    rospy.loginfo("%s started" % node_name)
    print("Started", node_name)

    image_capture_node = ImageRepublishNode()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down", node_name)
