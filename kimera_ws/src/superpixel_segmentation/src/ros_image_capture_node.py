#!/usr/bin/env python
import numpy as np
import argparse
from pathlib import Path
import rospy
import message_filters
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class ImageCaptureNode(object):
    def __init__(self) -> None:
        super().__init__()
        parser = argparse.ArgumentParser()
        args = self.parse_arguments(parser)
        self.bridge = CvBridge()

        self.image_topic = args.image_topic
        self.depth_topic = args.depth_topic
        self.seman_topic = args.semantics_topic
        self.mrcnn_topic = args.maskrcnn_topic
        self.supix_topic = args.superpixel_topic

        self.data_root = Path(args.save_folder)
        self.data_root.mkdir()
        
        self.image_folder = self.data_root / "image"
        self.depth_folder = self.data_root / "depth"
        self.seman_folder = self.data_root / "seman"
        self.mrcnn_folder = self.data_root / "mrcnn"
        self.supix_folder = self.data_root / "supix"

        self.image_folder.mkdir()
        self.depth_folder.mkdir()
        self.seman_folder.mkdir()
        self.mrcnn_folder.mkdir()
        self.supix_folder.mkdir()

        image_sub = message_filters.Subscriber(self.image_topic, Image, queue_size=50)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image, queue_size=50)
        seman_sub = message_filters.Subscriber(self.seman_topic, Image, queue_size=50)
        mrcnn_sub = message_filters.Subscriber(self.mrcnn_topic, Image, queue_size=50)
        supix_sub = message_filters.Subscriber(self.supix_topic, Image, queue_size=50)

        self.rate = rospy.Rate(1 / args.save_interval)
        ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, depth_sub, seman_sub, mrcnn_sub, supix_sub],
            1,
            slop=0.2,
        )
        ts.registerCallback(self.img_callback)

    def parse_arguments(self, parser):
        # Parse arguments for global params
        parser.add_argument(
            "--image_topic",
            type=str,
            default="/habitat/rgb/image_raw",
            help="Where the streaming RGB images are received over ROS",
        )
        parser.add_argument(
            "--depth_topic",
            type=str,
            default="/habitat/depth/image_raw",
            help="The topic where the robot depth images are published",
        )
        parser.add_argument(
            "--semantics_topic",
            type=str,
            default="/habitat/semantics/image_raw",
            help="The topic where the semantics images are published",
        )
        parser.add_argument(
            "--maskrcnn_topic",
            type=str,
            default="/semantics/semantic_image",
            help="The topic where the semantics images are published",
        )
        parser.add_argument(
            "--superpixel_topic",
            type=str,
            default="/semantics/superpixel",
            help="We are going to publish the superpixel segmented images over this topic",
        )
        parser.add_argument(
            "--camera_config",
            type=str,
            default="../config/calib_habitat.yml",
            help="The camera parameters of the virtual camera that renders the image",
        )
        parser.add_argument(
            "--save_folder",
            type=str,
            default="../assets/images",
            help="The directory where the resulting images will be stored uder separate folders.",
        )
        parser.add_argument(
            "--save_interval",
            type=int,
            default=1,
            help="The camera parameters of the virtual camera that renders the image",
        )

        args = parser.parse_args()

        return args

    def img_callback(self, image_msg, depth_msg, seman_msg, mrcnn_msg, supix_msg):
        print(f"Headers: {image_msg.header.stamp} {depth_msg.header.stamp} {seman_msg.header.stamp} {mrcnn_msg.header.stamp} {supix_msg.header.stamp}")

        try:
            cv_image: np.ndarray = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            cv_depth: np.ndarray = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            cv_seman: np.ndarray = self.bridge.imgmsg_to_cv2(seman_msg, "bgr8")
            cv_mrcnn: np.ndarray = self.bridge.imgmsg_to_cv2(mrcnn_msg, "bgr8")
            cv_supix: np.ndarray = self.bridge.imgmsg_to_cv2(supix_msg, "bgr8")

        except CvBridgeError as e:
            print("CvBridge parse exception: ", e)
            return

        cv_depth = (cv_depth / cv_depth.max()) * 255.

        image_pth = self.image_folder / f"{image_msg.header.stamp}.png"
        depth_pth = self.depth_folder / f"{image_msg.header.stamp}.png"
        seman_pth = self.seman_folder / f"{image_msg.header.stamp}.png"
        mrcnn_pth = self.mrcnn_folder / f"{image_msg.header.stamp}.png"
        supix_pth = self.supix_folder / f"{image_msg.header.stamp}.png"

        cv2.imwrite(str(image_pth), cv_image)
        cv2.imwrite(str(depth_pth), cv_depth)
        cv2.imwrite(str(seman_pth), cv_seman)
        cv2.imwrite(str(mrcnn_pth), cv_mrcnn)
        cv2.imwrite(str(supix_pth), cv_supix)

if __name__ == '__main__':

    rospy.init_node("image_capture_node")
    node_name = rospy.get_name()

    rospy.loginfo("%s started" % node_name)
    print("Started", node_name)

    image_capture_node = ImageCaptureNode()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down", node_name)
