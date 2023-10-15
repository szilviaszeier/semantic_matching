

import threading
import numpy as np
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_DEVICE_ID'] = '0'
import argparse
import yaml

import trimesh
import pyrender
print("PYRENDER",pyrender.__version__)
import cv2

from .utils import *
from .CameraConfig import CameraConfig

class TMeshRenderer(threading.Thread):

    def __init__(self, args):
        super().__init__()
        args = args
        self.mesh_path = args["mesh_path"]
        self.dataset_path = remove_trailing(args["recordings_path"])
        self.poses_path = self.dataset_path + '/poses/'
        self.images_path = self.dataset_path + '/imgs/'
        self.visualization = args["visualization"]

        print('Loading mesh:', self.mesh_path)
        self.trimesh = trimesh.load(self.mesh_path)

        with open(args["camera_config"]) as file:
            camera_config_yml = yaml.load(file, Loader=yaml.FullLoader)
            self.camera_config = CameraConfig(camera_config_yml)

        self.event_start = threading.Event()
        self.event_end = threading.Event()

    def run(self):
        self.mesh = pyrender.Mesh.from_trimesh(self.trimesh)
        self.scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
        self.scene.add(self.mesh)

        self.render_camera = self.camera_config.render_camera
        self.renderer = pyrender.offscreen.OffscreenRenderer(
            self.camera_config.width,
            self.camera_config.height)
        

        trans = np.array([2.0, 1.0, 0.0])
        quat = np.array([0.707, 0.0, 0.0, 0.707])
        pose = ros_to_opengl(quat, trans)
        self.camera_node = self.scene.add(self.render_camera, pose=pose)

        print('Finished scene initialization')

        if self.visualization:
            pyrender.Viewer(self.scene, viewport_size=self.camera_config.image_shape())

        while True:
            while not self.event_start.wait():
                pass
            self._render_ros_transform(self.trans, self.rot)
            self.event_start.clear()
            self.event_end.set()


    def render(self, pose):

        self.scene.set_pose(self.camera_node, pose=pose)

        rendered_img, renderd_depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.FLAT) #4096
        rendered_img = rendered_img[::-1]

        return rendered_img, renderd_depth


    def _render_ros_transform(self, trans, rot):

        quat_np = [rot[3], rot[0], rot[1], rot[2]]  # [qw, qx, qy, qz]
        pose = ros_to_opengl(quat_np, trans)

        self.rendered_img, self.rendered_depth = self.render(pose)
        self.rendered_img = cv2.cvtColor(self.rendered_img, cv2.COLOR_BGR2RGB)
        self.rendered_img = cv2.flip(self.rendered_img, 0)

    
    def render_ros_transform(self, trans, rot, secs, depth = False):
        self.trans = trans
        self.rot = rot
        self.event_start.set()
        while not self.event_end.wait():
            pass
        self.event_end.clear()
        if depth:
            print(f"Rendered RGB and D, with trans: {trans} and rot {rot} at {secs}")
            return self.rendered_img.copy(), self.rendered_depth.copy()

        else:
            print(f"Rendered image, with trans: {trans} and rot {rot} at {secs}")
            return self.rendered_img.copy()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Parse arguments for global params
    parser.add_argument('--mesh_path', type=str, default='../meshes/kinect_2cm_rgb.ply',
                        help='The semantic or RGB mesh, that provides the labels for the views')
    parser.add_argument('--camera_config', type=str, default='../config/calib_habitat.yml',
                        help='The camera parameters of the virtual camera that renders the image')
    parser.add_argument("--visualization", default=True, type=str2bool, nargs='?',
                        help="If we would like to open a viewer with the loaded model first")
    
    args = vars(parser.parse_args())
    meshrenderer = TMeshRenderer(args)

    trans = np.array([0.0, 0.0, 0.0])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    pose = ros_to_opengl(quat, trans)

    rendered_img, _ = meshrenderer.render(pose)

    trans = np.array([2.0, 1.0, 0.0])
    quat = np.array([0.707, 0.0, 0.0, 0.707])
    pose = ros_to_opengl(quat, trans)

    rendered_img2, _ = meshrenderer.render(pose)

    numpy_horizontal = np.hstack((rendered_img, rendered_img2))
    numpy_horizontal = cv2.resize(numpy_horizontal, (0, 0), fx=0.5, fy=0.5)
    numpy_horizontal = cv2.cvtColor(numpy_horizontal, cv2.COLOR_BGR2RGB)
    numpy_horizontal = cv2.flip(numpy_horizontal, 0)

    cv2.imshow('Rendered Images', numpy_horizontal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
