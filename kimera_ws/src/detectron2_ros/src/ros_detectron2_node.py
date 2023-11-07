#!/usr/bin/env python
import time

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from predictor import VisualizationDemo
from util import categories, setup_cfg


class Detectron2Node(object):
    def __init__(self) -> None:
        super().__init__()
        args = rospy.get_param("detectron2_node")
        image_topic = args["image_topic"]
        semantic_topic = args["semantic_topic"]
        self.bridge = CvBridge()
        cfg = setup_cfg(args)

        self.segmenter = VisualizationDemo(cfg, categories)

        self.image_sub = rospy.Subscriber(
            image_topic, Image, queue_size=1, callback=self.img_callback
        )
        self.semantic_pub = rospy.Publisher(semantic_topic, Image, queue_size=1)

    def img_callback(self, img_msg):
        img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

        start_time = time.time()
        predictions, visualized_output = self.segmenter.run_on_image(img)

        rospy.loginfo(
            f'detected {len(predictions["instances"]) if "instances" in predictions else 0} instances in {time.time() - start_time:.2f}s'
        )
        try:
            semantic_msg = self.bridge.cv2_to_imgmsg(visualized_output, "bgr8")
            semantic_msg.header = img_msg.header
            self.semantic_pub.publish(semantic_msg)

        except CvBridgeError as e:
            print("CvBridge publish exception: ", e)


if __name__ == "__main__":
    rospy.init_node("detectron2_node")
    node_name = rospy.get_name()

    rospy.loginfo("%s started" % node_name)
    print("Started", node_name)

    detectron2_node = Detectron2Node()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down", node_name)
