#!/usr/bin/env python
import numpy as np
from typing import Dict, Tuple
import rospy
import message_filters
import tf
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import yaml
from collections import defaultdict

from fast_slic.avx2 import SlicAvx2
from utils import transform_to_matrix, create_supix_mode_img, create_supix_mode_img_camera
from utils import downsample_point_cloud, cluster_point_cloud_with_traceback, determine_corresponding_labels, merge_point_cloud_clusters, paint_point_cloud, traceback_point_cloud, slic3d, determine_corresponding_labels_ # dbscan imports
from voxelgrid import VoxelGrid

import rendering

import time
import open3d as o3d

class SuperpixelNode(object):

    def __init__(self) -> None:
        super().__init__()

        rospy.init_node("superpixel_node")
        node_name = rospy.get_name()

        rospy.loginfo("%s started" % node_name)

        args = rospy.get_param("superpixel_segmentation")
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()
        self.num_components = 200
        self.slic = SlicAvx2(num_components=self.num_components, compactness=10)

        self.method = args["method"]
        self.pos_difference = args["pos_difference"]
        self.depth_scale = args["depth_scale"]
        self.gt_ratio = args["gt_ratio"]

        self.sensor_frame = args["sensor_frame"]
        self.parent_frame = args["parent_frame"]

        self.image_topic = args["image_topic"]
        self.depth_topic = args["depth_topic"]
        self.superpixel_topic = args["superpixel_topic"]

        self.voxel_size = args["voxel_size"]
        with open(args["camera_config"]) as file:
            camera_config_yml = yaml.load(file, Loader=yaml.FullLoader)
            self.camera_config = rendering.CameraConfig(camera_config_yml)

        self.pos_mat = self.preprocess_position_matrix()

        self.mesh_path = args["mesh_path"]
        self.voxel_grid = VoxelGrid(self.mesh_path, voxel_size=self.voxel_size)
        #self.voxel_grid.project_colors_down() # Currently not needed

        self.meshrenderer = rendering.TMeshRenderer(args)
        self.meshrenderer.start()
        image_sub = message_filters.Subscriber(self.image_topic, Image, queue_size=1)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image, queue_size=1)
        if self.method == "baseline":
            self.semantic_img_fn = self.create_superpixel_semantic_img_baseline
            self.pixel_grid = self.voxel_to_pixel_grid()
        elif self.method == "camera":
            self.semantic_img_fn = self.create_superpixel_semantic_img_camera_plane
        else:
            self.mesh_o3d = o3d.io.read_triangle_mesh(self.mesh_path)
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh_o3d)

            self.mesh_scene = o3d.t.geometry.RaycastingScene()
            _ = self.mesh_scene.add_triangles(mesh)
            if self.method == "3d_supix":
                self.semantic_img_fn = self.create_superpixel_semantic_img_3d_supix
            elif self.method == "dbscan":
                self.semantic_img_fn = self.create_superpixel_semantic_img_dbscan
            elif self.method == "3d_neighbor":
                self.semantic_img_fn = self.create_superpixel_semantic_img_3d_neighbor
            elif self.method == "3d_slic":
                self.semantic_img_fn = self.create_superpixel_semantic_img_3d_slic

        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 1, slop=0.1)
        ts.registerCallback(self.img_callback)

        self.superpixel_image_pub = rospy.Publisher(self.superpixel_topic, Image, queue_size=5)

        rospy.spin()

    def preprocess_position_matrix(self):
        pos_mat = np.zeros((self.camera_config.height, self.camera_config.width, 3))
        for i in range(self.camera_config.height):
            for j in range(self.camera_config.width):
                pos_mat[i,j] = np.array([1,
                                        (self.camera_config.K[0,2] - j),
                                        (self.camera_config.K[1,2] - i)])
        return pos_mat
 
    def pcd_from_depth(self, curr_transform, pos_mat, depth_mat, depth_scale=1):
        '''
        Calculate the 3D positions of each pixel in the frame if the depth measure is valid (i.e. greater than 1e-4) to obtain a point cloud
        '''
        pos_3d = np.multiply(pos_mat, depth_mat / depth_scale)
        fx, fy = self.camera_config.focal_length()
        pos_3d[:,:,1] /= fx
        pos_3d[:,:,2] /= fy
        transform = np.zeros((pos_3d.shape[0], pos_3d.shape[1], 4, 4))
        transform[:,:] = np.eye(4)
        transform[:,:,:3,3] = pos_3d
        curr_transform = curr_transform[np.newaxis, np.newaxis, ...] # Create new axis for height and width
        curr_transform = np.repeat(np.repeat(curr_transform, pos_3d.shape[1], axis=1), pos_3d.shape[0], axis=0) # Transform into shape H x W x 4 x 4
        pos_3d = (curr_transform @ transform)[:,:,:3,3]
        return pos_3d
    
    def voxel_to_pixel_grid(self):
        '''
        Project the voxel grid of the input mesh onto the floor to create a pixel grid.
        '''
        x_range = self.voxel_grid.max_x - self.voxel_grid.min_x + 1
        y_range = self.voxel_grid.max_y - self.voxel_grid.min_y + 1
        z_range = self.voxel_grid.max_z - self.voxel_grid.min_z + 1
        ceiling_size = int(z_range * 0.2)
        pixel_grid = np.zeros((x_range, y_range, 3))
        for i in range(x_range):
            for j in range(y_range):
                cat = (0,0,0)
                for k in range(z_range - ceiling_size, 0, -1):
                    coord = i + self.voxel_grid.min_x, j + self.voxel_grid.min_y, k + self.voxel_grid.min_z
                    if self.voxel_grid.is_filled_voxel(coord):
                        prov_cat = self.voxel_grid.get_color_voxel(coord)
                        # ceiling: SUNRGBD-(137, 28, 157), ADA20K-(255,190,190)
                        # floor: SUNRGBD-(54, 114, 113), ADA20K-(244,243,131)
                        if prov_cat != (255,190,190) and (cat == (0,0,0) or prov_cat != (244,243,131)):
                            cat = prov_cat

                pixel_grid[i,j] = cat
        return pixel_grid

    def compare_pcds(self, pcd1, pcd2):
        '''
        Compare point clouds, find where difference in position is too large.
        '''
        return np.linalg.norm(pcd1 - pcd2, axis=2) < self.pos_difference

    def create_superpixel_semantic_img_baseline(self, assignment:np.ndarray, pcd_mask:np.ndarray, rendered_sem:np.ndarray):
        ''' Using the point cloud mask and get semantic labels of areas with large distance from the floor projection (pixel grid).
        '''
        # (480, 640, 1), (480, 640, 3)
        start = time.time()

        semantic_img = np.where(pcd_mask[..., np.newaxis], rendered_sem[...,::-1], 0).astype(np.uint16)

        scaled_calc_pcd = (self.calculated_pcd // self.voxel_grid.voxel_size).astype(int)
        # (480, 640, 3) = 3d coord per pixel
        scaled_calc_pcd[..., 0] -= self.voxel_grid.min_x
        scaled_calc_pcd[..., 1] -= self.voxel_grid.min_y
        scaled_calc_pcd[..., 2] -= self.voxel_grid.min_z
        mesh_mask = (0 <= scaled_calc_pcd[..., 0]) & \
                    (scaled_calc_pcd[..., 0] < self.pixel_grid.shape[0]) & \
                    (0 <= scaled_calc_pcd[..., 1]) & \
                    (scaled_calc_pcd[..., 1] < self.pixel_grid.shape[1])
        far_pixel_indicies = np.where((~pcd_mask & mesh_mask)[..., np.newaxis], scaled_calc_pcd[..., :2], 0)

        far_pixel_colors = self.pixel_grid[far_pixel_indicies[..., 0].ravel(), far_pixel_indicies[..., 1].ravel(), :].reshape(semantic_img.shape)
        semantic_img[~pcd_mask] = far_pixel_colors[~pcd_mask]

        # most common color per superpixel
        superpixel_semantic_img = np.zeros(semantic_img.shape, dtype=np.uint16)
        create_supix_mode_img(semantic_img.reshape(-1, 3), assignment.astype(np.uint16).ravel(), superpixel_semantic_img.reshape(-1, 3))

        print(time.time() - start)

        return superpixel_semantic_img

    def create_superpixel_semantic_img_camera_plane(self, assignment:np.ndarray, pcd_mask:np.ndarray, rendered_sem:np.ndarray):
        '''
        Using the point cloud mask and the rendered semantic image, extend the superpixels.
        '''
        start = time.time()
        semantic_img = np.where(pcd_mask[..., np.newaxis], rendered_sem[...,::-1], 0).astype(np.uint16)

        # most common color per superpixel
        superpixel_semantic_img = np.zeros(semantic_img.shape, dtype=np.uint16)
        create_supix_mode_img_camera(semantic_img.reshape(-1, 3), assignment.astype(np.uint16).ravel(), pcd_mask.astype(np.uint16).ravel(), np.float32(self.gt_ratio), superpixel_semantic_img.reshape(-1, 3))

        print(time.time() - start)
        return superpixel_semantic_img

    def create_superpixel_semantic_img_3d_supix(self, assignment:np.ndarray, depth_img:np.ndarray):
        '''
        assignment: superpixel segmentation
        depth_img: the observed depth scaled to meters
        calculated_pcd: point cloud calculated from the observations (depth + pose)
        '''
        reduced_pcd = self.calculated_pcd.reshape(-1,3)[::5]
        reduced_assignment = assignment.reshape(-1, 1)[::5]
        reduced_depth = (depth_img / self.depth_scale).ravel()[::5]

        query_points = o3d.core.Tensor(reduced_pcd, dtype=o3d.core.Dtype.Float32)
        ans = self.mesh_scene.compute_closest_points(query_points)
        indices = np.asarray(self.mesh_o3d.triangles)[ans['primitive_ids'].numpy()][:,0]
        sem_img = np.asarray(self.mesh_o3d.vertex_colors)[indices]
        sem_img[reduced_depth<0.01] = 0

        superpixel_semantic_img = np.zeros(sem_img.shape, dtype=np.uint16)
        create_supix_mode_img((sem_img*255).astype(np.uint16), reduced_assignment.astype(np.uint16).ravel(), superpixel_semantic_img)

        _, unique_supix_ind = np.unique(reduced_assignment, return_index=True)
        supix_colors = superpixel_semantic_img[unique_supix_ind]
        superpixel_semantic_img = supix_colors[assignment]

        return superpixel_semantic_img

    def create_superpixel_semantic_img_dbscan(self, assignment:np.ndarray, depth_img:np.ndarray):
        original_pcd = o3d.geometry.PointCloud()
        original_pcd.points = o3d.utility.Vector3dVector(self.calculated_pcd.reshape(-1,3))
        original_pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(original_pcd.points), 3)))
        downsampled_pcd, downsample_traceback = downsample_point_cloud(original_pcd, voxel_size=0.04, return_trace=True)
        clustered_pcd_list, clustered_traceback = cluster_point_cloud_with_traceback(downsampled_pcd,
                                                                                        reduce_pcd_noise=False,
                                                                                        ransac_num_iterations=300,
                                                                                        max_n_planes=5)
        cluster_labels = determine_corresponding_labels(clustered_pcd_list, self.mesh_scene, self.mesh_o3d)

        merged_pcd = merge_point_cloud_clusters(len(downsampled_pcd.points), clustered_pcd_list, clustered_traceback,
                                                cluster_labels)
        paint_point_cloud(downsampled_pcd, merged_pcd)
        traceback_point_cloud(original_pcd, downsampled_pcd, downsample_traceback)

        return (np.asarray(original_pcd.colors)*255).reshape(*self.camera_config.image_shape(), 3)

    def create_superpixel_semantic_img_3d_neighbor(self, assignment:np.ndarray, depth_img:np.ndarray):
        '''
        depth_img: the observed depth scaled to meters
        calculated_pcd: point cloud calculated from the observations (depth + pose)
        '''
        pcd = self.calculated_pcd.reshape(-1,3)

        query_points = o3d.core.Tensor(pcd, dtype=o3d.core.Dtype.Float32)
        ans = self.mesh_scene.compute_closest_points(query_points)
        indices = np.asarray(self.mesh_o3d.triangles)[ans['primitive_ids'].numpy()][:,0]
        sem_img = np.asarray(self.mesh_o3d.vertex_colors)[indices]*255
        sem_img[(depth_img / self.depth_scale).ravel()<0.01] = 0

        return sem_img.reshape(*self.camera_config.image_shape(), 3)

    def create_superpixel_semantic_img_3d_slic(self, assignment:np.ndarray, depth_img:np.ndarray, n_sv=512, voxel_size=0.1):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.calculated_pcd.reshape(-1,3))
        pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(pcd.points), 3)))

        pcd_down, traceback, sv_colors = slic3d(pcd, n_sv=n_sv, voxel_size=voxel_size, run_traceback=False)

        traceback_point_cloud(pcd, pcd_down, traceback)
        #return pcd, np.asarray(pcd.colors).reshape(*img_shape[:2], 3)
        
        cluster_labels = determine_corresponding_labels_(pcd_down, sv_colors, self.mesh_scene, self.mesh_o3d, update_pcd_color=True)
        #print(cluster_labels)
        traceback_point_cloud(pcd, pcd_down, traceback)
        return (np.asarray(pcd.colors)*255).reshape(*self.camera_config.image_shape()[:2], 3)


    def create_superpixel_semantic_img_voxel(self, cv_depth: np.ndarray, pos_3d: np.ndarray, assignment: np.ndarray) -> np.ndarray:
        # pos_3d + assignment -> semantic labeled

        # mapping => cluster : {color : number of occurance}
        label_dict: Dict[int, Dict[Tuple[int,int,int], int]] = defaultdict(lambda: defaultdict(int))

        for i in range(self.camera_config.height):
            for j in range(self.camera_config.width):
                if cv_depth[i,j] and self.voxel_grid.is_filled_point(pos_3d[i,j]):
                    label_dict[assignment[i,j]][self.voxel_grid.get_color_point(pos_3d[i,j])] += 1

        # mapping => cluster : most common color
        max_dict: Dict[int, Tuple[int,int,int]] = {d : max(set(label_dict[d]), key=label_dict[d].get) for d in label_dict}

        k = np.array(list(label_dict.keys()))
        v = np.array(list(max_dict.values()))
        mapping_ar = np.zeros((self.num_components + 1, 3), dtype=v.dtype)
        mapping_ar[k] = v
        superpixel_semantic_img = mapping_ar[assignment]
        return superpixel_semantic_img

    def img_callback(self, img_msg, depth_msg):
        start = time.time()
        try:
            cv_image:np.ndarray = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            cv_depth:np.ndarray = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except CvBridgeError as e:
            print("CvBridge parse exception: ", e)
            return

        assignment:np.ndarray = self.slic.iterate(cv_image)

        try:
            # self.listener.waitForTransform(
            #     self.parent_frame,
            #     self.sensor_frame,
            #     img_msg.header.stamp, rospy.Duration(10))
            
            trans, rot = self.listener.lookupTransform(
                self.parent_frame,
                self.sensor_frame,
                img_msg.header.stamp)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("Transform exception: ", e)
            return

        rot_rev = [rot[3]] + rot[:3] # rearrange quaternions into the correct order

        rendered_sem, rendered_depth = self.meshrenderer.render_ros_transform(trans, rot, img_msg.header.stamp.secs, depth=True)
        
        curr_transform = transform_to_matrix(rot_rev, trans)
        
        pos_start = time.time()
        self.calculated_pcd = self.pcd_from_depth(curr_transform, self.pos_mat, cv_depth[:,:,np.newaxis], depth_scale=self.depth_scale)
        pos_end = time.time()
        pos_time = pos_end - pos_start
        rendered_pcd = self.pcd_from_depth(curr_transform, self.pos_mat, rendered_depth[:,:,np.newaxis])
        pcd_mask = self.compare_pcds(self.calculated_pcd, rendered_pcd)

        sem_start = time.time()
        if self.method in ["3d_supix", "3d_neighbor", "dbscan", "3d_slic"]:
            superpixel_semantic_img = self.semantic_img_fn(assignment, cv_depth)
        else:
            superpixel_semantic_img = self.semantic_img_fn(assignment, pcd_mask, rendered_sem)
        sem_end = time.time()
        sem_time = sem_end - sem_start
        try:
            superpixel_semantic_img_msg = self.bridge.cv2_to_imgmsg(superpixel_semantic_img.astype('uint8'), "rgb8")
            superpixel_semantic_img_msg.header = img_msg.header
            self.superpixel_image_pub.publish(superpixel_semantic_img_msg)

        except CvBridgeError as e:
            print("CvBridge publish exception: ", e)

        
        end = time.time()
        print(f"Segmentation elapsed time:{(end-start):1.4f} pos:{pos_time:1.4f} sem:{sem_time:1.4f}")


if __name__ == '__main__':
    try:
        superpixel_node = SuperpixelNode()
    except rospy.ROSInterruptException:
        pass
