import numpy as np
import math
import argparse
from typing import Dict, Tuple, Union, List
from numba import guvectorize, uint16, float32
import copy

import open3d as o3d
from skimage.segmentation import slic

# TODO this is here only for testing. This needs to be centralised and imported in all files:
# habitat / traj_sim etc.
CLASS_COLORS = [(0,0,0), (119,119,119), (244,243,131), (255,190,190), (54,114,113), (255,150,255),
                (0,163,33), (150,255,0), (255,180,10), (150,255,255), (0,0,176), 
                (24,209,255), (152,163,55), (70,72,115), (87,64,34), (193,195,234),
                (192,79,212), (70,72,115), (52,57,131), (137,28,157), 
                (255,69,0)]

COLORS_ARRAY = np.array([[x[0], x[1], x[2]] for x in CLASS_COLORS], dtype=np.uint16)

COLOR_TO_IND_DICT: Dict[Tuple[int,int,int], int] = dict(zip(CLASS_COLORS, np.arange(len(CLASS_COLORS))))

transformation_quat = np.array([0.5, 0.5, -0.5, -0.5]) # [qw, qx, qy, qz]

def map_labels(labels, label_map, dtype=None):
    """Map the given label matrix by the specified mapping dictionary.

    Parameters
    ----------
    labels : np.ndarray
        Label matrix consisting of hashable objects.
    label_map : dict of {hashable: object}
        Dictionary defining the mapping of the labels.
    dtype : data-type, optional
        Data type of the mapping target, if not specified use the type of the first value of `label_map`,
        by default None.

    Returns
    -------
    np.ndarray
        Mapped `labels` by the `label_map`.
    """
    if 0 == len(label_map):
        return labels.reshape((*labels.shape, 1))

    # assert type(label_map.keys()[0]) <= labels.dtype
    label_ids = np.unique(labels)  # sorted(label_map.keys())
    tmp = np.asarray(list(label_map.values())[0])

    dim = tmp.shape
    assert len(dim) <= 1
    if len(dim) == 0:
        dim = 1
    else:
        dim = dim[0]
    assert 0 < dim

    if dtype is None:
        dtype = tmp.dtype

    lookup_table = np.empty((int(label_ids[-1] + 1), dim), dtype=dtype)
    for label_id in label_ids:
        print(label_id)
        if label_id in label_map:
            lookup_table[label_id] = label_map[label_id]
        else:
            print(label_id)
            lookup_table[label_id] = label_id
    relabeled = lookup_table[labels.reshape(-1), :].reshape((*labels.shape, dim))
    return relabeled

@guvectorize([(uint16[:,:], uint16[:], uint16[:,:])], '(n,c),(n)->(n,c)', nopython=True)
def create_supix_mode_img(array, assignment, result):
    num_components = assignment.max() + 1
    # compile time needed, may seek a better way by passing some parameter
    col_to_ind = {
        (0,0,0) : 0,
        (119,119,119) : 1,
        (244,243,131) : 2,
        (255,190,190) : 3,
        (54,114,113) : 4,
        (255,150,255) : 5,
        (0,163,33) : 6,
        (150,255,0) : 7,
        (255,180,10) : 8,
        (150,255,255) : 9,
        (0,0,176) : 10,
        (24,209,255) : 11,
        (152,163,55) : 12,
        (70,72,115) : 13,
        (87,64,34) : 14,
        (193,195,234) : 15,
        (192,79,212) : 16,
        (70,72,115) : 17,
        (52,57,131) : 18,
        (137,28,157) : 19,
        (255,69,0) : 20
    }

    #col_to_ind2 = dict([(1000000*k[0] + 1000*k[1] + k[2], v) for k,v in list(col_to_ind.items())])

    for comp in range(num_components):
        mask = assignment == comp
        a = array[mask]
        
        b = [col_to_ind[(r,g,b)] for r,g,b in a if (r,g,b) in col_to_ind]
        # TODO: Speedup fails due to anti-aliasing when rendering image from input reconstruction
        #tmp = (1000000*a[:, 0] + 1000*a[:, 1] + a[:, 2]).astype(int)
        #b = map_labels(tmp, col_to_ind2)[:, 0]

        mode_ind = np.bincount(b).argmax() if len(b) != 0 else 0
        result[mask] = COLORS_ARRAY[mode_ind]

#@guvectorize([(uint16[:,:], uint16[:], uint16[:], float32, uint16[:,:])], '(n,c),(n),(n),()->(n,c)', nopython=True)
def create_supix_mode_img_camera(array, assignment, pcd_mask, gt_ratio, result):
    num_components = assignment.max() + 1
    # compile time needed, may seek a better way by passing some parameter
    col_to_ind = {
        (0,0,0) : 0,
        (119,119,119) : 1,
        (244,243,131) : 2,
        (255,190,190) : 3,
        (54,114,113) : 4,
        (255,150,255) : 5,
        (0,163,33) : 6,
        (150,255,0) : 7,
        (255,180,10) : 8,
        (150,255,255) : 9,
        (0,0,176) : 10,
        (24,209,255) : 11,
        (152,163,55) : 12,
        (70,72,115) : 13,
        (87,64,34) : 14,
        (193,195,234) : 15,
        (192,79,212) : 16,
        (70,72,115) : 17,
        (52,57,131) : 18,
        (137,28,157) : 19,
        (255,69,0) : 20
    }

    col_to_ind2 = dict([(1000000*k[0] + 1000*k[1] + k[2], v) for k,v in list(col_to_ind.items())])

    for comp in range(num_components):
        mask = assignment == comp
        extendable = ((pcd_mask & mask).sum() / mask.sum()) > gt_ratio
        if extendable:
            a = array[mask]
            
            #b = [col_to_ind[(r,g,b)] for r,g,b in a if (r,g,b) in col_to_ind]
            # TODO: Speedup fails due to anti-aliasing when rendering image from input reconstruction
            tmp = (1000000*a[:, 0] + 1000*a[:, 1] + a[:, 2]).astype(np.int64)
            b = map_labels(tmp, col_to_ind2, np.int64)[:, 0]

            mode_ind = np.bincount(b).argmax() if len(b) != 0 else 0
            result[mask] = COLORS_ARRAY[mode_ind]
        else:
            result[mask] = array[mask]

## DBSCAN method functions
def downsample_point_cloud(pcd, voxel_size=0.02, return_trace=True):
    if not return_trace:
        return pcd.voxel_down_sample(voxel_size)
    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()
    voxel_down_pcd, _, traceback_list = pcd.voxel_down_sample_and_trace(voxel_size, min_bound, max_bound, False)
    return voxel_down_pcd, traceback_list

def find_planes(pcd: o3d.geometry.PointCloud,
                max_n_planes: int = 50, ransac_distance_threshold: float = 0.03, ransac_num_iterations: int = 3000,
                dbscan_eps: float = 0.1, dbscan_min_points: int = 10,
                colors: np.ndarray = None, return_traceback: bool = False) \
        -> Union[Tuple[List[o3d.geometry.PointCloud], o3d.geometry.PointCloud],
                 Tuple[List[o3d.geometry.PointCloud], o3d.geometry.PointCloud, np.ndarray, List[np.ndarray]]]:
    # RANSAC with Euclidean clustering
    if colors is None:
        colors = np.array(generate_random_colors(max_n_planes)) / 255.

    # segment_models = []
    planes = []
    rest = copy.deepcopy(pcd)
    np.random.seed(0)
    if return_traceback:
        traceback_list = []
        remaining_traceback = np.ones(len(pcd.points), dtype=np.bool)
    for i in range(max_n_planes):
        if len(rest.points) <= 3:
            break
        segment_model, inliers = rest.segment_plane(distance_threshold=ransac_distance_threshold,
                                                    ransac_n=3, num_iterations=ransac_num_iterations, seed=0)
        # segment_models.append(segment_model)
        segment = rest.select_by_index(inliers)
        labels = np.array(segment.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points))
        candidates = [len(np.where(labels == j)[0]) for j in np.unique(labels)]
        best_candidate = np.unique(labels)[np.where(candidates == np.max(candidates))[0]][0]

        # Stop if there's too much noise
        if best_candidate == -1:
            break

        rest = rest.select_by_index(inliers, invert=True) + segment.select_by_index(
            list(np.where(labels != best_candidate)[0]))
        '''
        if best_candidate == -1:
            print("Skip plane:", i)
            planes[i] = None
            continue
        '''
        if return_traceback:
            curr_traceback = np.zeros_like(remaining_traceback)
            if 0 == len(traceback_list):
                with fancy_index_view(curr_traceback, inliers) as barview:
                    barview[labels == best_candidate] = True
            else:
                tmp = curr_traceback[remaining_traceback]
                with fancy_index_view(tmp, inliers) as barview:
                    barview[labels == best_candidate] = True
                curr_traceback[remaining_traceback] = tmp
            traceback_list.append(np.where(curr_traceback)[0])
            remaining_traceback[curr_traceback] = False
        segment = segment.select_by_index(list(np.where(labels == best_candidate)[0]))
        segment.paint_uniform_color(colors[i])

        planes.append(segment)
    if return_traceback:
        return planes, rest, traceback_list, remaining_traceback
    return planes, rest

def generate_random_color() -> List[int]:  # TODO: speed it up
    """Generates a random color.
    Returns
    -------
    list of int
        A randomly generated color.
    """
    return list(np.random.choice(range(256), size=3))


def generate_random_colors(n_colors: int, sorted: bool = False, include_alpha: bool = False) -> np.ndarray:
    """Randomly generates a list of unique colors.
    Parameters
    ----------
    n_colors : int
        Number of unique colors to generate.
    Returns
    -------
    list of list of int
        A sorted list of randomly generated unique colors.
    """
    colors = np.unique([generate_random_color() for _ in range(n_colors)], axis=0)
    while len(colors) < n_colors:
        colors = np.concatenate((colors,
                                 np.asarray([generate_random_color()
                                             for _ in range(n_colors - len(colors))])), axis=0)
        colors = np.unique(colors, axis=0)
    colors = colors.astype(np.uint8)

    if not sorted:
        rng = np.random.default_rng()
        rng.shuffle(colors, axis=0)
    if include_alpha:
        colors = np.concatenate((colors, np.full((n_colors, 1), 255, dtype=np.uint8)), axis=1)
    return colors

def dbscan_clustering(pcd: o3d.geometry.PointCloud,
                      dbscan_eps: float = 0.05, dbscan_min_points: int = 5,
                      colors: np.ndarray = None, remove_noise: bool = True,
                      traceback_vector: np.ndarray = None):
    # TODO: remove small clusters
    # Euclidean clustering with DBSCAN: rest.select_by_index(inliers, invert=True)
    pcd = copy.deepcopy(pcd)

    labels = np.array(pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points))
    max_label = labels.max()
    # print(labels.shape, labels.min(), labels.max(), np.unique(labels, return_counts=True))

    if colors is None:
        colors = np.array(generate_random_colors(max(0, max_label) + 1)) / 255.
    colors = colors[labels]
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # Remove noise
    if remove_noise:
        labels = np.where(labels < 0)[0]
        print("#LABELS", len(labels))
        if traceback_vector is not None:
            tmp = traceback_vector[traceback_vector]
            with fancy_index_view(tmp, labels) as barview:
                barview[::] = False
            traceback_vector[traceback_vector] = tmp
            #traceback_vector[traceback_vector][labels] = False
        pcd = pcd.select_by_index(labels, invert=True)

    if traceback_vector is not None:
        return pcd, traceback_vector
    return pcd

import contextlib

@contextlib.contextmanager
def fancy_index_view(arr, inds):
    # Source: https://stackoverflow.com/a/36868161
    # create copy from fancy inds
    arr_copy = arr[inds]

    # yield 'view' (copy)
    yield arr_copy

    # after context, save modified data
    arr[inds] = arr_copy

def split_point_clouds(pcd, traceback_vector: np.ndarray = None):
    ucolors, uindices, ucounts = np.unique(pcd.colors, return_inverse=True, return_counts=True, axis=0)
    uindices_ord = np.argsort(uindices)

    label_indices_list = []
    start_idx = 0
    for c in ucounts:
        label_indices_list.append(uindices_ord[start_idx:start_idx + c])
        start_idx += c

    pcds = [pcd.select_by_index(indices) for indices in label_indices_list]
    if traceback_vector is not None:
        tracback_list = []
        for indices in label_indices_list:
            curr_traceback = np.zeros_like(traceback_vector)

            tmp = curr_traceback[traceback_vector]
            with fancy_index_view(tmp, indices) as barview:
                barview[::] = True
            curr_traceback[traceback_vector] = tmp
            #curr_traceback[traceback_vector][indices] = True
            tracback_list.append(np.where(curr_traceback)[0])
        return pcds, tracback_list
    return pcds

def cluster_point_cloud_with_traceback(pcd: o3d.geometry.PointCloud, reduce_pcd_noise: bool = False, n_points: int = -1,
                                       ransac_distance_threshold: float = 0.03, ransac_num_iterations: int = 3000,
                                       max_n_planes: int = 50,
                                       ransac_dbscan_eps: float = 0.1, ransac_dbscan_min_points: int = 10,
                                       dbscan_eps: float = 0.1, dbscan_min_points: int = 10):
    if reduce_pcd_noise:
        raise NotImplementedError()
        # pcd = reduce_noise(pcd, apply_smoothing=False, n_points=n_points)
    planes, rest, traceback_list, traceback_vector = \
        find_planes(pcd, ransac_distance_threshold=ransac_distance_threshold,
                    ransac_num_iterations=ransac_num_iterations, max_n_planes=max_n_planes,
                    dbscan_eps=ransac_dbscan_eps, dbscan_min_points=ransac_dbscan_min_points,
                    return_traceback=True)
    #print(np.count_nonzero(traceback_vector))
    if len(rest.points) == 0:
        return planes, traceback_list
    rest, traceback_vector = dbscan_clustering(rest, dbscan_eps=dbscan_eps, dbscan_min_points=dbscan_min_points,
                                               traceback_vector=traceback_vector, remove_noise=False)
    #print(np.count_nonzero(traceback_vector))
    rest, rest_traceback_list = split_point_clouds(rest, traceback_vector=traceback_vector)
    pcd_clustered = planes + rest
    traceback_list += rest_traceback_list
    return pcd_clustered, traceback_list

def filter_mesh_by_pcd(pcd, scene, mesh):
    query_points = o3d.core.Tensor(np.asarray(pcd.points), dtype=o3d.core.Dtype.Float32)
    ans = scene.compute_closest_points(query_points)
    indices = np.asarray(mesh.triangles)[ans['primitive_ids'].numpy()][:,0]
    selected_mesh_indices = list(indices)
    filtered_mesh = mesh.select_by_index(selected_mesh_indices)
    return filtered_mesh
    
def determine_corresponding_labels(pcd_list, scene, mesh, update_pcd_color=False):
    labels = np.empty((len(pcd_list), 3), dtype=np.float64)
    for idx, pcd in enumerate(pcd_list):
        if np.all(np.asarray(pcd.colors)[0] == 0):
            color = np.asarray([0, 0, 0], dtype=np.float64)
        else:
            filtered_mesh = filter_mesh_by_pcd(pcd, scene, mesh)
            colors = np.asarray(filtered_mesh.vertex_colors)
            ucolors, ucounts = np.unique(colors, return_counts=True, axis=0)
            color = ucolors[np.argmax(ucounts)] if 0 < len(ucounts) else np.zeros(3, dtype=np.float64)
        labels[idx] = color
        if update_pcd_color:
            np.asarray(pcd.colors)[:] = color 
    return labels
    
def merge_point_cloud_clusters(n_points, pcd_list, traceback, labels):
    np_points = np.zeros((n_points, 3), dtype=np.float64)
    np_colors = np.zeros((n_points, 3), dtype=np.float64)

    for color, tb, pcd in zip(labels, traceback, pcd_list):
        np_points[tb] = np.asarray(pcd.points)
        np_colors[tb] = color

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)
    pcd.colors = o3d.utility.Vector3dVector(np_colors)
    return pcd

def paint_point_cloud(pcd_to_paint, pcd_guide):
    def determine_sort_order(pcd):
        tmp = np.asarray(pcd.points)
        tmp = 10000*tmp[:, 0] + 100*tmp[:, 1] + tmp[:, 2]
        return np.argsort(tmp)

    sort_keys_new = determine_sort_order(pcd_guide)
    sort_keys_orig = determine_sort_order(pcd_to_paint)
    
    tmp_colors_new = np.asarray(pcd_guide.colors)
    tmp_colors_orig = np.asarray(pcd_to_paint.colors)
    tmp_colors_orig[sort_keys_orig] = tmp_colors_new[sort_keys_new]
    np.asarray(pcd_to_paint.colors)[::] = tmp_colors_orig[::]

def traceback_point_cloud(original_pcd, downsampled_pcd, downsampled_traceback):
    downsampled_pcd_colors = np.asarray(downsampled_pcd.colors)
    original_pcd_colors = np.asarray(original_pcd.colors)
    for i, tb in enumerate(downsampled_traceback):
        original_pcd_colors[tb] = downsampled_pcd_colors[i]
## End DBSCAN functions

## Start 3D SLIC functions
def slic3d(pcd, n_sv=512, voxel_size=0.1, run_traceback=True):
    def points_to_coord(points, origin, voxel_size):
        return (points - pcd_down.get_center()) / voxel_size

    pcd_down, traceback = downsample_point_cloud(pcd, voxel_size=voxel_size, return_trace=True)

    # Get the shape, the indices and the colors of the voxel grid
    indices = points_to_coord(np.asarray(pcd_down.points), pcd_down.get_center(), voxel_size)
    min_idx, max_idx = indices.min(axis=0), indices.max(axis=0)
    # min_idx, max_idx = points_to_coord(np.stack((pcd.get_min_bound(), pcd.get_max_bound())), pcd_down.get_center(), voxel_size)
    grid_shape = np.round(max_idx - min_idx + 1).astype(int)
    indices = np.round(indices - min_idx).astype(int)
    colors = np.asarray(pcd_down.colors)

    # Create the voxel grid
    np_vg = np.zeros((*grid_shape, 3), dtype=np.uint8)
    np_vg[indices[:, 0], indices[:, 1], indices[:, 2]] = colors  # np.round(colors*255)
    np_vg_mask = np.zeros(grid_shape, dtype=bool)
    np_vg_mask[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    # Run slic3D
    vx_labels = slic(np_vg, n_segments=n_sv, mask=np_vg_mask, max_num_iter=5, compactness=0.1)
    #print(pcd_down.points, n_sv, vx_labels.max() + 1, len(np.unique(vx_labels)))
    n_sv = vx_labels.max() + 1
    label_colors = generate_random_colors(n_sv + 1) / 255
    #print(label_colors)
    new_colors = label_colors[vx_labels[indices[:, 0], indices[:, 1], indices[:, 2]]]
    pcd_down.colors = o3d.utility.Vector3dVector(new_colors)

    if run_traceback:
        traceback_point_cloud(pcd, pcd_down, traceback)
        return pcd, label_colors[:n_sv]
    return pcd_down, traceback, label_colors[:n_sv]

def determine_corresponding_labels_(pcd, label_colors, scene, mesh, update_pcd_color=False):
    np_points = np.asarray(pcd.points)
    np_colors = np.asarray(pcd.colors)
    labels = np.empty((len(label_colors), 3), dtype=np.float64)
    for idx, label_color in enumerate(label_colors):
        #if not np.all(label_color == 0):
        #coords = np.where((np.sum((np_colors - color)**2, axis=-1) < 4.))
        coords = np.where(np.sum((np_colors == label_color), axis=-1) == 3)
        points = np_points[coords]
        #print(points.shape)
        colors = filter_mesh_by_pcd_(points, scene, mesh)
        #filtered_mesh = filter_mesh_by_pcd_(points, scene, mesh)
        #colors = np.asarray(filtered_mesh.vertex_colors)
        #print(len(points), len(colors), colors)
        #print(colors.shape)
        #print('-----------')
        ucolors, ucounts = np.unique(colors, return_counts=True, axis=0)
        color = ucolors[np.argmax(ucounts)] if 0 < len(ucounts) else np.zeros(3, dtype=np.float64)
        labels[idx] = color
        if update_pcd_color:
            np_colors[coords] = color
    if update_pcd_color:
        pcd.colors = o3d.utility.Vector3dVector(np_colors)
        
    return labels

def filter_mesh_by_pcd_(np_points, scene, mesh):
    query_points = o3d.core.Tensor(np_points, dtype=o3d.core.Dtype.Float32)
    ans = scene.compute_closest_points(query_points)
    indices = np.asarray(mesh.triangles)[ans['primitive_ids'].numpy()][:, 0]
    #print(len(np_points), len(indices), indices)
    return np.asarray(mesh.vertex_colors)[indices]
    #return np_mesh_colors[indices]
    #selected_mesh_indices = list(indices)
    #filtered_mesh = mesh.select_by_index(selected_mesh_indices)
    #print(len(np_points), len(indices), filtered_mesh)
    #return filtered_mesh
## End 3D SLIC functions

def remove_trailing(path):

    path1 = path.rstrip('/')
    path2 = path1.rstrip('\\')

    return path2


def quaternion_multiply(Q0, Q1):
    """
    Multiplies two quaternions.

    Input
    :param Q0: A 4 element array containing the first quaternion (q01,q11,q21,q31)
    :param Q1: A 4 element array containing the second quaternion (q02,q12,q22,q32)

    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33)

    """
    # Extract the values from Q0
    w0 = Q0[0]
    x0 = Q0[1]
    y0 = Q0[2]
    z0 = Q0[3]

    # Extract the values from Q1
    w1 = Q1[0]
    x1 = Q1[1]
    y1 = Q1[2]
    z1 = Q1[3]

    # Computer the product of the two quaternions, term by term
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    # Create a 4 element array containing the final quaternion
    final_quaternion = np.array([Q0Q1_w, Q0Q1_x, Q0Q1_y, Q0Q1_z])

    # Return a 4 element array containing the final quaternion (q02,q12,q22,q32)
    return final_quaternion

def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if mag2 > tolerance:
        mag = math.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return np.array(v)

def transform_to_matrix(Q, trans):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (qw,qx,qy,qz)

    Output
    :return: A 4x4 element matrix representing the full 3D rotation matrix
             and zero translation.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    hom_matrix = np.eye(4)
    hom_matrix[:3, :3] = rot_matrix
    hom_matrix[:3, 3] = trans

    return hom_matrix


def ros_to_opengl(quat, trans):

    # OpenGL: RH, x - right, y - up, z - back
    # ROS: RH, x - forward, y - left, z - up

    opengl_quat = quaternion_multiply(normalize(quat), transformation_quat)

    opengl_pose = transform_to_matrix(opengl_quat, trans)

    return opengl_pose


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')