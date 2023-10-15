#!/usr/bin/env python
import rospy
import cv2
import tf
from sensor_msgs.msg import Image
import message_filters
from cv_bridge import CvBridge, CvBridgeError

from meshrenderer import MeshRenderer
from utils import *


class RendererNode(object):

    def __init__(self):
        super().__init__()
        
        args = rospy.get_param("rendering")
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        self.sensor_frame = args["sensor_frame"]
        self.parent_frame = args["parent_frame"]
        self.image_topic = args["image_topic"]
        self.depth_topic = args["depth_topic"]
        self.rendered_topic = args["rendered_topic"]
        self.rendered_overlayed_topic = args["rendered_overlayed_topic"]
        self.rendered_depth_topic = args["rendered_depth_topic"]
        self.visualization = args["visualization"]

        self.meshrenderer = MeshRenderer(args)

        image_sub = message_filters.Subscriber(self.image_topic, Image, queue_size=10)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image, queue_size=10)
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 10, slop=0.1)
        ts.registerCallback(self.img_callback)

        self.rendered_image_pub = rospy.Publisher(self.rendered_topic, Image, queue_size=5)
        self.overlayed_image_pub = rospy.Publisher(self.rendered_overlayed_topic, Image, queue_size=5)
        self.rendered_depth_pub = rospy.Publisher(self.rendered_depth_topic, Image, queue_size=5)

    def img_callback(self, img_msg, depth_msg):

        try:

            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")

        except CvBridgeError as e:
            print("CvBridge parse exception: ", e)
            return

        try:
            self.listener.waitForTransform(self.parent_frame,
                                           self.sensor_frame,
                                           img_msg.header.stamp, rospy.Duration(10))
            (trans, rot) = self.listener.lookupTransform(self.parent_frame,
                                                         self.sensor_frame,
                                                         img_msg.header.stamp)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("Transform exception: ", e)
            return

        rendered_img, rendered_depth = self.meshrenderer.render_ros_transform(trans, rot, img_msg.header.stamp.secs, depth=True)

        try:
            rendered_msg = self.bridge.cv2_to_imgmsg(rendered_img, "bgr8")
            rendered_msg.header = img_msg.header
            self.rendered_image_pub.publish(rendered_msg)

            rendered_depth_msg = self.bridge.cv2_to_imgmsg(rendered_depth)
            rendered_depth_msg.header = img_msg.header
            self.rendered_depth_pub.publish(rendered_depth_msg)

            if self.visualization:

                overlay  = cv2.addWeighted(rendered_img, 0.5, cv_image, 0.5, 0)
                rendered_overlayed_msg = self.bridge.cv2_to_imgmsg(overlay)
                rendered_overlayed_msg.header = img_msg.header
                self.overlayed_image_pub.publish(rendered_overlayed_msg)


        except CvBridgeError as e:
            print("CvBridge publish exception: ", e)


if __name__ == '__main__':

    rospy.init_node("mesh_renderer_node")
    node_name = rospy.get_name()

    rospy.loginfo("%s started" % node_name)
    print("Started", node_name)

    render_node = RendererNode()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down", node_name)

