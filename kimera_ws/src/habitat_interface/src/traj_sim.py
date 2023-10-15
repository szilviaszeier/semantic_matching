#!/usr/bin/env python
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import cv2
import pickle
import copy
import os.path
import json

from skimage.filters import gaussian


import habitat_sim

import rosbag
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image as rosImage
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage
import geometry_msgs.msg as rosgeo
import rospy
import tf2_msgs.msg as tf2msg
import tf2_ros

#from rospy_message_converter import message_converter
#from rospy_message_converter import json_message_converter

from utils import make_cfg, transform_habitat_to_ROS, CLASS_DICT

class InteractiveSimulator(object):

    def __init__(self):
        rospy.init_node("habitat_node")
        args = rospy.get_param("habitat_simulation")
        args["width"] = rospy.get_param("habitat_simulation/sim/width")
        args["height"] = rospy.get_param("habitat_simulation/sim/height")
        self.init_simulator(args)
        self.init_ros(args)
        if not self.replay_mode:
            self.navigator_subsriber = rospy.Subscriber("navigator/action", String, self.update_action)

        self.agent_pose = []
        self.camera_pose = []

        self.run_simulation()

    def init_simulator(self, args):
        sim_settings = rospy.get_param("habitat_simulation/sim")
        self.sensor_height = sim_settings["sensor_height"]
        random.seed(sim_settings["seed"])

        cfg = make_cfg(sim_settings, args["mesh_path"], args["depth_noise_multiplier"])

        self.sim = habitat_sim.Simulator(cfg)
        self.sim.seed(sim_settings["seed"])
        self.sim.config.sim_cfg.allow_sliding = True

        # set new initial state
        self.sim.initialize_agent(agent_id=0)
        self.agent = self.sim.agents[0]
        
        ##
        self.replay_mode:bool = args["replay_mode"]
        self.save_bag:bool = args["save_bag"]
        self.save_files:bool = args["save_files"]
        if self.save_files:
            self.save_files_dir = args["save_files_dir"]
            os.makedirs(self.save_files_dir)

        self.show_sensor_images:bool = args["show_cameras"]
        self.target_fps:int = args["target_fps"]

        self.instanceid2class = self._generate_label_map(self.sim.semantic_scene)
        self.class2color = self._generate_object_dict(self.sim.semantic_scene)
        self.map_to_class_labels = np.vectorize(
            lambda x: self.class2color.get(
                self.instanceid2class.get(x, 0), 
                CLASS_DICT["object"]
            )
        )

        self.gaussian_sigma:float = args["gaussian_sigma"]
        self.motion_blur_weight:float = args["motion_blur_weight"]
        self.norm_factor = self.normalize(self.motion_blur_weight)
        self.prev_rgb:List[np.ndarray] = []
        self.prev_depth:List[np.ndarray] = []
        self.prev_semantic:List[np.ndarray] = []

        self.queue_size:int = args["queue_size"]

    def init_ros(self, args):

        self.compressed:bool = args["compressed"]
        self.output_agent_pose_name:str = args["output_agent_pose_name"]
        if self.save_bag:
            self.bag = rosbag.Bag(args["output_bag_name"], 'w')

        self.bridge = CvBridge()
        # Topic names
        self.rgb_topic_name:str = args["image_topic_name"]
        self.depth_topic_name:str = args["depth_topic_name"]
        self.semantic_topic_name:str = args["semantic_topic_name"]

        self.rgb_info_topic_name:str = self.rgb_topic_name.rsplit('/', 1)[0] + '/camera_info'
        self.depth_info_topic_name:str = self.depth_topic_name.rsplit('/', 1)[0] + '/camera_info'
        self.semantic_info_topic_name:str = self.semantic_topic_name.rsplit('/', 1)[0] + '/camera_info'

        self.compressed_image_topic_name:str = args["compressed_image_topic_name"]
        self.compressed_depth_topic_name:str = args["compressed_depth_topic_name"]
        self.compressed_semantic_topic_name:str = args["compressed_semantic_topic_name"]

        self.world_frame:str = args["parent_frame"]
        self.robot_frame:str = args["robot_frame"]

        self.publish_tf:bool = args["publish_tf"]

        # Setup camera info msg
        hfov:float = float(self.agent.agent_config.sensor_specifications[0].hfov) * np.pi / 180.
        self.height:int = self.agent.agent_config.sensor_specifications[0].resolution[0]
        self.width:int = self.agent.agent_config.sensor_specifications[0].resolution[1]
        focal_length:float = (self.width / 2) / np.tan(hfov / 2.0) #264.587

        self.camera_info_msg:CameraInfo = CameraInfo()
        self.camera_info_msg.header.frame_id = "habitat_camera_frame"
        self.camera_info_msg.width = args["width"]
        self.camera_info_msg.height = args["height"]
        self.camera_info_msg.K = [focal_length, 0, self.width / 2.0, 0, focal_length, self.height/2.0, 0, 0, 1]
        self.camera_info_msg.D = [0, 0, 0, 0, 0]
        self.camera_info_msg.R = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
        self.camera_info_msg.P = [focal_length, 0, self.width / 2.0, 0, 0, focal_length, self.height/2.0, 0, 0, 0, 1.0, 0]
        self.camera_info_msg.distortion_model = 'plumb_bob'

        # Publishers
        self.rgb_pub = rospy.Publisher(self.rgb_topic_name, rosImage, queue_size=self.queue_size)
        self.depth_pub = rospy.Publisher(self.depth_topic_name, rosImage, queue_size=self.queue_size)
        self.semantic_pub = rospy.Publisher(self.semantic_topic_name, rosImage, queue_size=self.queue_size)

        self.rgb_info_pub = rospy.Publisher(self.rgb_info_topic_name, CameraInfo, queue_size=self.queue_size)
        self.depth_info_pub = rospy.Publisher(self.depth_info_topic_name, CameraInfo, queue_size=self.queue_size)
        self.semantic_info_pub = rospy.Publisher(self.semantic_info_topic_name, CameraInfo, queue_size=self.queue_size)

        self.tf_br_pub = tf2_ros.TransformBroadcaster()

    def run_simulation(self):
        # Initialization for motion blur
        #self.init_motion_blur()

        rate = rospy.Rate(self.target_fps)

        if self.replay_mode:
            replay_states = self.load_poses(self.output_agent_pose_name)
            for i, state in enumerate(replay_states):
                # state.sensor_states = state.velocity
                # state.velocity = np.array([0., 0., 0.])
                # state.angular_velocity = np.array([0., 0., 0.])
                # state.force = np.array([0., 0., 0.])
                # state.torque = np.array([0., 0., 0.])
                #print(state)
                # for sensor_i in state.sensor_states:
                #     state.sensor_states[sensor_i].position[1] = state.position[1] + self.sensor_height
                self.agent.set_state(state, infer_sensor_states=True)
                self.render()
                rospy.loginfo(f"State: {i:04d} / {len(replay_states)}")
                rate.sleep()

            print("Replay finished.")

        else:
            loop_active = True
            self.action = "stay"
            while loop_active and not rospy.is_shutdown():
                if self.action != "exit":
                    self.sim.step(self.action)
                    self.render()
                    rate.sleep()
                else:
                    loop_active = False
            with open(self.output_agent_pose_name, "wb") as fp:
                pickle.dump(self.agent_pose, fp)

        if self.save_bag:
            self.bag.close()
        self.sim.reset()

    def update_action(self, action:String):
        self.action = action.data

    def render(self):
        # render observation
        observation = self.sim.get_sensor_observations()

        # Color
        rgb = observation["color_sensor"]
        
        # Semantics
        # 0.2s
        #640x480 = 307200
        semantic = observation["semantic_sensor"]
        semantic = np.asarray(self.map_to_class_labels(semantic))
        semantic = np.stack((semantic[0], semantic[1], semantic[2]), axis=2)
        semantic = semantic.astype(np.uint8)

        # Depth
        depth = observation["depth_sensor"]
        # TODO: depth conversion might be necessary here
        #if self.args.encoding == 32:
        #    depth = np.float32(depth)
        #else:
        #    depth = np.uint16(depth)

        # TODO: add motion blur
        #rgb, depth, semantic = self.motion_blur(rgb, depth, semantic)

        timestamp:rospy.Time = rospy.Time.now()
        agent_state = self.agent.get_state()
        self.agent_pose.append(agent_state)

        tf_msg = self.create_tf_msg(agent_state, timestamp)

        rgb_msg, rgb_info = self.create_img_msg(rgb[:, :, :3], "rgb8", timestamp)
        depth_msg, depth_info = self.create_img_msg(depth, "passthrough", timestamp)
        semantic_msg, semantic_info = self.create_img_msg(semantic, "rgb8", timestamp)

        self.publish_msgs(
            rgb_msg=rgb_msg, depth_msg=depth_msg, semantic_msg=semantic_msg,
            rgb_info=rgb_info, depth_info=depth_info, semantic_info=semantic_info,
            tf_msg=tf_msg)

        if self.save_bag:
            try:
                self.write_to_bag(
                    rgb_msg=rgb_msg, depth_msg=depth_msg, semantic_msg=semantic_msg,
                    rgb_info=rgb_info, depth_info=depth_info, semantic_info=semantic_info,
                    tf_msg=tf_msg)
            except Exception as e:
                #print(e)
                return
        '''
        if self.save_files:
            try:
                self.write_to_files(timestamp=timestamp,
                    rgb_msg=rgb_msg, depth_msg=depth_msg, semantic_msg=semantic_msg,
                    rgb_info=rgb_info, depth_info=depth_info, semantic_info=semantic_info,
                    agent_state=agent_state)
            except Exception as e:
                #print(e)
                return
        '''

        if self.show_sensor_images:
            self.show_sensors(rgb, depth, semantic)

    def show_sensors(self, rgb:np.ndarray, depth:np.ndarray, semantic:np.ndarray):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) / 255.
        semantic_bgr = cv2.cvtColor(semantic , cv2.COLOR_RGB2BGR) / 255.
        depth3 = cv2.cvtColor(depth / 10.0, cv2.COLOR_GRAY2BGR)  ## to scale under float 1.0
        dst = np.hstack((bgr, depth3, semantic_bgr))

        cv2.imshow("Rendered sensors", cv2.resize(dst, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(1)

    def create_img_msg(self, img:np.ndarray, encoding:str, timestamp:rospy.Time) -> Tuple[rosImage, tf2msg.TFMessage]:
        info_msg = copy.deepcopy(self.camera_info_msg)
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding)
        img_msg.header.frame_id = self.camera_info_msg.header.frame_id
        img_msg.header.stamp = timestamp

        info_msg.header.stamp = timestamp
        return img_msg, info_msg

    def publish_msgs(self,
            rgb_msg:rosImage, depth_msg:rosImage, semantic_msg:rosImage,
            rgb_info:tf2msg.TFMessage, depth_info:tf2msg.TFMessage, semantic_info:tf2msg.TFMessage,
            tf_msg:tf2msg.TFMessage):

        if self.publish_tf:
            for transform in tf_msg.transforms:
                self.tf_br_pub.sendTransform(transform)

        self.rgb_pub.publish(rgb_msg)
        self.rgb_info_pub.publish(rgb_info)
        self.depth_pub.publish(depth_msg)
        self.depth_info_pub.publish(depth_info)
        self.semantic_pub.publish(semantic_msg)
        self.semantic_info_pub.publish(semantic_info)

    def write_to_bag(self,
            rgb_msg:rosImage, depth_msg:rosImage, semantic_msg:rosImage,
            rgb_info:tf2msg.TFMessage, depth_info:tf2msg.TFMessage, semantic_info:tf2msg.TFMessage,
            tf_msg:tf2msg.TFMessage):

        self.bag.write(self.rgb_info_topic_name, rgb_info)
        self.bag.write(self.depth_info_topic_name, depth_info)
        self.bag.write(self.semantic_info_topic_name, semantic_info)

        if not self.compressed:
            self.bag.write(self.rgb_topic_name, rgb_msg)
            self.bag.write(self.depth_topic_name, depth_msg)
            self.bag.write(self.semantic_topic_name, semantic_msg)
        else:
            image_color = CompressedImage()
            image_color.format = "jpeg"
            image_color.data = np.array(cv2.imencode('.jpg', rgb_msg.data[:, :, :3])[1]).tostring()

            image_depth = CompressedImage()
            image_depth.format = ""
            image_depth.data = np.array(cv2.imencode('.jpg', depth_msg.data)[1]).tostring()

            image_semantic = CompressedImage()
            image_semantic.format = "jpeg"
            image_semantic.data = np.array(cv2.imencode('.jpg', semantic_msg.data)[1]).tostring()

            self.bag.write(self.compressed_image_topic_name, image_color)
            self.bag.write(self.compressed_depth_topic_name, image_depth)
            self.bag.write(self.compressed_semantic_topic_name, image_semantic)

        if self.publish_tf:
            for transform in tf_msg.transforms:
                tfmsg_pub = tf2msg.TFMessage()
                tfmsg_pub.transforms.append(transform)
                self.bag.write('/tf', tfmsg_pub)
    '''
    def write_to_files(self,
            timestamp:rospy.Time,
            rgb_msg:rosImage, depth_msg:rosImage, semantic_msg:rosImage,
            rgb_info:tf2msg.TFMessage, depth_info:tf2msg.TFMessage, semantic_info:tf2msg.TFMessage,
            agent_state):

        filename = str(timestamp.to_nsec())

        # Export color image
        try:
            if rgb_msg.encoding == '8UC1' or rgb_msg.encoding == 'mono8':
                rgb_image = self._cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='passthrough')
            elif rgb_msg.encoding == '16UC1':
                rgb_image = self._cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='passthrough')
                # we could save this on 8bits if we scale the typical depth range (let's say 15m) into 0-255 range. Scale factor 255.0/4096.0
            elif rgb_msg.encoding == '32FC1':
                print("TODO: float depth export is not yet implemented")
                # need scaling by 65535
                assert(False)
            elif rgb_msg.encoding == 'rgb8' or rgb_msg.encoding == 'rgba8':
                rgb_image = self._cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            else:
                print("Error: unknown image encoding: {0}".format(rgb_msg.encoding))
                return
            cv2.imwrite(os.path.join(self.save_files_dir, filename + ".rgb.png"),rgb_image)            
        except CvBridgeError as e:
            print(e)
            return

        # Export depth image
        try:
            if depth_msg.encoding == '8UC1' or depth_msg.encoding == 'mono8':
                depth_image = self._cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            elif rgb_msg.encoding == '16UC1':
                depth_image = self._cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
                # we could save this on 8bits if we scale the typical depth range (let's say 15m) into 0-255 range. Scale factor 255.0/4096.0
            elif depth_msg.encoding == '32FC1':
                print("TODO: float depth export is not yet implemented")
                # need scaling by 65535
                assert(False)
            elif depth_msg.encoding == 'rgb8' or depth_msg.encoding == 'rgba8':
                depth_image = self._cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='bgr8')
            else:
                print("Error: unknown image encoding: {0}".format(depth_msg.encoding))
                return
            cv2.imwrite(os.path.join(self.save_files_dir, filename + ".depth.png"), depth_image)        
        except CvBridgeError as e:
            print(e)
            return

        # Export semantic image
        try:
            if semantic_msg.encoding == '8UC1' or semantic_msg.encoding == 'mono8':
                semantic_image = self._cv_bridge.imgmsg_to_cv2(semantic_msg, desired_encoding='passthrough')
            elif semantic_msg.encoding == '16UC1':
                semantic_image = self._cv_bridge.imgmsg_to_cv2(semantic_msg, desired_encoding='passthrough')
                # we could save this on 8bits if we scale the typical depth range (let's say 15m) into 0-255 range. Scale factor 255.0/4096.0
            elif semantic_msg.encoding == '32FC1':
                print("TODO: float depth export is not yet implemented")
                # need scaling by 65535
                assert(False)
            elif semantic_msg.encoding == 'rgb8' or semantic_msg.encoding == 'rgba8':
                semantic_image = self._cv_bridge.imgmsg_to_cv2(semantic_msg, desired_encoding='bgr8')
            else:
                print("Error: unknown image encoding: {0}".format(semantic_msg.encoding))
                return
            cv2.imwrite(os.path.join(self.save_files_dir, filename + ".semantic.png"), semantic_image)           
        except CvBridgeError as e:
            print(e)
            return


        # Export camera intrinsics
        try:
            camera_info_dict = message_converter.convert_ros_message_to_dictionary(rgb_info)
            with open(os.path.join(self.save_files_dir, filename + '.info.json'), 'w') as f:
                json.dump(camera_info_dict, f, sort_keys=True, indent=4)
        except Exception as e:
            print(e)
            return

        # Export pose
        try:            
            pose = transform_to_matrix(Q = agent_state.rotation, trans = agent_state.position)            
            with open(os.path.join(self.save_files_dir, filename + '.pose.txt'), 'w') as f:
                 np.savetxt(f, pose)
        except Exception as e:
            print(e)
            return
        
        print("Successfully exported frame " + filename)
    '''

    def create_tf_msg(self, agent_state:habitat_sim.AgentState, timestamp:rospy.Time) -> tf2msg.TFMessage:

        tf_msg = tf2msg.TFMessage()

        ### BASE FRAME TRANSFORMATION
        base_trf = self.create_transform(
            timestamp=timestamp, agent_state=agent_state,
            header_frame_id=self.world_frame, child_frame_id=self.robot_frame,
            sensor_name="base")
        tf_msg.transforms.append(base_trf)

        ### COLOR TRANSFORMATION
        rgb_trf = self.create_transform(
            timestamp=timestamp, agent_state=agent_state,
            header_frame_id=self.robot_frame, child_frame_id="habitat_rgb_frame",
            sensor_name="color_sensor")
        tf_msg.transforms.append(rgb_trf)
        
        ### DEPTH TRANSFORMATION
        depth_trf = self.create_transform(
            timestamp=timestamp, agent_state=agent_state,
            header_frame_id=self.robot_frame, child_frame_id="habitat_depth_frame",
            sensor_name="depth_sensor")
        tf_msg.transforms.append(depth_trf)

        ### SEMANTIC TRANSFORMATION
        semantic_trf = self.create_transform(
            timestamp=timestamp, agent_state=agent_state,
            header_frame_id=self.robot_frame, child_frame_id="habitat_semantic_frame",
            sensor_name="semantic_sensor")
        tf_msg.transforms.append(semantic_trf)

        return tf_msg

    def create_transform(
            self, timestamp:rospy.Time, agent_state:habitat_sim.AgentState,
            header_frame_id:str, child_frame_id:str, sensor_name:str) -> rosgeo.TransformStamped:

        transform = rosgeo.TransformStamped()

        transform.header.stamp = timestamp
        transform.header.frame_id = header_frame_id
        transform.child_frame_id = child_frame_id

        if sensor_name == "base":
            trans = agent_state.position
            rot = agent_state.rotation
        else:
            trans = agent_state.sensor_states[sensor_name].position - agent_state.position
            rot = agent_state.rotation.inverse() * agent_state.sensor_states[sensor_name].rotation

        transform.transform.translation.x = float(trans[0])
        transform.transform.translation.y = float(trans[1])
        transform.transform.translation.z = float(trans[2])
        transform.transform.rotation.x = rot.x
        transform.transform.rotation.y = rot.y
        transform.transform.rotation.z = rot.z
        transform.transform.rotation.w = rot.w
        transform = transform_habitat_to_ROS(transform)

        return transform

    def load_poses(self, filename):
        if os.path.isfile(filename):
            return np.load(filename, allow_pickle=True)
        else:
            raise FileNotFoundError(f"Could not load agent states from non existant file: {self.output_agent_pose_name}")

    def _generate_label_map(self, scene) -> Dict[int, str]:
        instance_id_to_name: Dict[int, str] = {}
        for obj in scene.objects:
            if obj and obj.category:
                obj_id = int(obj.id.split("_")[-1])
                instance_id_to_name[obj_id] = obj.category.name()

        return instance_id_to_name

    def _generate_object_dict(self, scene) -> Dict[str, Tuple[int,int,int]]:

        object_dict = CLASS_DICT.copy()

        object_dict['base-cabinet'] = CLASS_DICT['cabinet']
        object_dict['cabinet'] = CLASS_DICT['cabinet']
        object_dict['wall-cabinet'] = CLASS_DICT['cabinet']
        object_dict['wardrobe'] = CLASS_DICT['cabinet']
        object_dict['desk'] = CLASS_DICT['table']
        object_dict['blinds'] = CLASS_DICT['curtain']
        object_dict['shower-stall'] = CLASS_DICT['wall']
        object_dict["bin"] = CLASS_DICT["trashcan"]
        object_dict["tv-screen"] = CLASS_DICT["monitor"]

        return object_dict

    def motion_blur(self, rgb:np.ndarray, depth:np.ndarray, semantic:np.ndarray)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        self.prev_rgb.append(gaussian(rgb, sigma=self.gaussian_sigma, multichannel=True, preserve_range=True))
        self.prev_depth.append(gaussian(depth, sigma=self.gaussian_sigma, multichannel=True, preserve_range=True))
        self.prev_semantic.append(gaussian(semantic, sigma=self.gaussian_sigma, multichannel=True, preserve_range=True))

        self.prev_rgb = self.prev_rgb[-10:]
        self.prev_depth = self.prev_depth[-10:]
        self.prev_semantic = self.prev_semantic[-10:]

        _rgb = (sum([1 / self.motion_blur_weight ** i * self.prev_rgb[i] for i in range(9, -1, -1)]) / self.norm_factor).astype(np.uint8)
        _depth = sum([1 / self.motion_blur_weight ** i * self.prev_depth[i] for i in range(9, -1, -1)]) / self.norm_factor
        # Not goot to average the labels! You get average color and not one of the labels!
        _semantic = semantic #(sum([1 / self.motion_blur_weight ** i * self.prev_semantic[i] for i in range(9, -1, -1)]) / self.norm_factor).astype(np.uint8)

        return _rgb, _depth, _semantic

    def normalize(self, c):
        return sum([1/c**i for i in range(10)]) 

    def init_motion_blur(self):
        for i in range(10):
            self.sim.step("stay")
            # render observation
            observation = self.sim.get_sensor_observations()
            rgb = observation["color_sensor"]
            semantic = observation["semantic_sensor"]

            semantic = np.asarray(self.map_to_class_labels(semantic))
            semantic = np.stack((semantic[0], semantic[1], semantic[2]), axis=2)
            semantic = semantic.astype(np.uint8)

            depth = observation["depth_sensor"]

            self.prev_rgb.append(gaussian(rgb, sigma=self.gaussian_sigma, multichannel=True, preserve_range=True))
            self.prev_depth.append(gaussian(depth, sigma=self.gaussian_sigma, multichannel=True, preserve_range=True))
            self.prev_semantic.append(gaussian(semantic, sigma=self.gaussian_sigma, multichannel=True, preserve_range=True))


if __name__ == "__main__":
    InteractiveSimulator()