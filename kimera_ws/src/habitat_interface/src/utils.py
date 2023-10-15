import copy
from typing import Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt
# function to display the topdown map
from PIL import Image

import habitat_sim

import cv2
import geometry_msgs.msg as rosgeo

# TODO import from mask_rcnn_ros/src/sunrgbd.py class
#CLASS_NAMES = ['BG', 'bed', 'books', 'ceiling', 'chair', 'floor',
#                'furniture', 'objects', 'picture', 'sofa', 'table',
#                'tv', 'wall', 'window']
#
#FOCUSED_NAMES = ['BG', 'bed', 'books', 'ceiling', 'chair', 'floor',
#                'furniture', 'objects', 'picture', 'sofa', 'table',
#                'tv', 'wall', 'window']
#
#CLASS_COLORS = [(0, 0, 0), (119, 119, 119), (244, 243, 131),
#                (137, 28, 157), (150, 255, 255), (54, 114, 113),
#                (0, 0, 176), (255, 69, 0), (87, 112, 255), (0, 163, 33),
#                (255, 150, 255), (255, 180, 10), (101, 70, 86),
#                (38, 230, 0)]

CLASS_NAMES = ['wall', 'floor', 'ceiling', 'chair', 'table',
                'window', 'curtain', 'picture', 'bed', 'sofa',
                'pillow', 'monitor', 'sink', 'trashcan', 'toilet', 
                'refrigerator', 'oven', 'bathtub', 'cabinet',
                  'object']

CLASS_COLORS = [(119,119,119), (244,243,131), (255,190,190), (54,114,113), (255,150,255),
                (0,163,33), (150,255,0), (255,180,10), (150,255,255), (0,0,176), 
                (24,209,255), (152,163,55), (70,72,115), (87,64,34), (193,195,234),
                (192,79,212), (70,72,115), (52,57,131), (137,28,157), 
                (255,69,0)]

CLASS_DICT: Dict[str, Tuple[int,int,int]] = dict(zip(CLASS_NAMES, CLASS_COLORS))

# Configure Sim Settings
def make_cfg(settings, mesh_path, depth_noise_multiplier:float = 0.0) -> habitat_sim.Configuration:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = mesh_path
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    color_sensor_spec.hfov = 100.0
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [settings["depth_offset"], settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    depth_sensor_spec.hfov = 100.0
    if depth_noise_multiplier > 0.0:
        depth_sensor_spec.noise_model = "RedwoodDepthNoiseModel"
        depth_sensor_spec.noise_model_kwargs = dict(noise_multiplier = depth_noise_multiplier)
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    semantic_sensor_spec.hfov = 100.0
    sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.04)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=1.5)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=1.5)
        ),
        "move_backward": habitat_sim.agent.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=0.04)
        ),
        "stay": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=0.7)
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=0.7)
        ),
        "move_up": habitat_sim.agent.ActionSpec(
            "move_up", habitat_sim.agent.ActuationSpec(amount=0.02)
        ),
        "move_down": habitat_sim.agent.ActionSpec(
            "move_down", habitat_sim.agent.ActuationSpec(amount=0.02)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def transform_habitat_to_ROS(habitat_trans):

    ### Habitat -> ROS
    ### X -> -Y
    ### Y -> Z
    ### Z -> -X

    ros_trans = rosgeo.TransformStamped()
    ros_trans.header = copy.deepcopy(habitat_trans.header)
    ros_trans.child_frame_id = habitat_trans.child_frame_id

    ros_trans.transform.translation.x = - habitat_trans.transform.translation.z
    ros_trans.transform.translation.y = - habitat_trans.transform.translation.x
    ros_trans.transform.translation.z = habitat_trans.transform.translation.y

    ros_trans.transform.rotation.x = - habitat_trans.transform.rotation.z
    ros_trans.transform.rotation.y = - habitat_trans.transform.rotation.x
    ros_trans.transform.rotation.z = habitat_trans.transform.rotation.y
    ros_trans.transform.rotation.w = habitat_trans.transform.rotation.w

    return ros_trans

