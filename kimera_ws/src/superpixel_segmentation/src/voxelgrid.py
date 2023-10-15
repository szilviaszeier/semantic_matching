from typing import Dict, Tuple
import os
import copy
from plyfile import PlyData
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


class VoxelGrid(object):
    def __init__(self, file_path, voxel_size:float = 0.05):
        self.file_path = file_path
        self.voxel_size = voxel_size
        self.voxels: Dict[Tuple[int,int,int], Tuple[int,int,int]] = {}
        self._process_file()
        self.set_bounds()

    def _process_file(self):
        assert(os.path.isfile(self.file_path))
        with open(self.file_path, 'rb') as f:
            plydata = PlyData.read(f)
            num_vertex = plydata['vertex'].count
            vertices = np.zeros(shape=[num_vertex, 3], dtype=np.float32)
            colors = np.zeros(shape=[num_vertex, 3], dtype=np.uint8)
            vertices[:,0] = plydata['vertex'].data['x']
            vertices[:,1] = plydata['vertex'].data['y']
            vertices[:,2] = plydata['vertex'].data['z']
            colors[:,0] = plydata['vertex'].data['red']
            colors[:,1] = plydata['vertex'].data['green']
            colors[:,2] = plydata['vertex'].data['blue']
        
        temp_dict = defaultdict(lambda: [])
        for i in range(num_vertex):
            ind = self.get_voxel_index(vertices[i])
            temp_dict[tuple(ind.tolist())].append(tuple(colors[i].tolist()))

        for k in temp_dict.keys():
            self.voxels[k] = max(set(temp_dict[k]), key=temp_dict[k].count)

    def set_bounds(self):
        self.min_x: int = np.min([vox[0] for vox in self.voxels.keys()])
        self.min_y: int = np.min([vox[1] for vox in self.voxels.keys()])
        self.min_z: int = np.min([vox[2] for vox in self.voxels.keys()])

        self.max_x: int = np.max([vox[0] for vox in self.voxels.keys()])
        self.max_y: int = np.max([vox[1] for vox in self.voxels.keys()])
        self.max_z: int = np.max([vox[2] for vox in self.voxels.keys()])

    def get_voxel_index(self, point:np.ndarray) -> np.ndarray:
        return (point // self.voxel_size).astype(int)

    def add_point(self, point:np.ndarray, color:Tuple[int,int,int]):
        if not self.is_filled_point(point):
            ind = self.get_voxel_index(point)
            self.add_voxel(tuple(ind.tolist()), color)
        else:
            print(f"Point {point} already filled!")
    
    def add_voxel(self, ind:Tuple[int,int,int], color:Tuple[int,int,int]):
        if not self.is_filled_voxel(ind):
            self.voxels[ind] = color
        else:
            print(f"Voxel {ind} already filled!")

    def remove_point(self, point:np.ndarray):
        ind = self.get_voxel_index(point)
        self.remove_voxel(ind)

    def remove_voxel(self, ind:Tuple[int,int,int]):
        try:
            del self.voxels[ind]
        except Exception:
            print("Voxel removal error occurred.")

    def is_empty(self):
        return len(self.voxels) == 0

    def is_filled_point(self, point:np.ndarray) -> bool:
        ind = self.get_voxel_index(point)
        return self.is_filled_voxel(tuple(ind.tolist()))

    def is_filled_voxel(self, ind:Tuple[int,int,int]) -> bool:
        return ind in self.voxels

    def filled_count(self):
        return len(self.voxels)

    def project_colors_down(self):
        original_vox = copy.deepcopy(list(self.voxels.keys()))
        for vox in original_vox:
            if vox[2] <= self.max_z - 10:
                x = vox[0]
                y = vox[1]
                z = vox[2] - 1

                while z >= self.min_z and not self.is_filled_voxel((x, y, z)):
                    self.add_voxel((x, y, z), self.voxels[vox])
                    z -= 1

        self.set_bounds()


    def get_color_point(self, point:np.ndarray) -> Tuple[int,int,int]:
        ind = self.get_voxel_index(point)
        return self.get_color_voxel(tuple(ind.tolist()))

    def get_color_voxel(self, ind:Tuple[int,int,int]) -> Tuple[int,int,int]:
        return self.voxels[ind]

    def print_bounds(self):
        print(f"Min corner: ({self.min_x:4d},{self.min_y:4d},{self.min_z:4d})")
        print(f"Max corner: ({self.max_x:4d},{self.max_y:4d},{self.max_z:4d})")

if __name__ == '__main__':
    mesh_path = '../meshes/habitat_frl_apartment_4_semantic_mesh.ply'
    vg = VoxelGrid(mesh_path, 0.1)
    print(f"Old voxel count: {vg.filled_count()}")

    vg.print_bounds()
    vg.project_colors_down()
    vg.print_bounds()

    print(f"New voxel count: {vg.filled_count()}")

    ax = plt.figure().add_subplot(projection='3d')

    arr = np.zeros(((vg.max_x-vg.min_x)+1, (vg.max_y-vg.min_y)+1, (vg.max_z-vg.min_z)+1))
    col_arr = np.zeros(((vg.max_x-vg.min_x)+1, (vg.max_y-vg.min_y)+1, (vg.max_z-vg.min_z)+1, 4), dtype=np.uint8)
    print(arr.shape)

    def calib(p:Tuple[int,int,int]) -> Tuple[int,int,int]:
        return (p[0]-vg.min_x, p[1]-vg.min_y, p[2]-vg.min_z)
    
    for pos, col in vg.voxels.items():
        arr[calib(pos)] = 1
        col_arr[calib(pos)] = np.array(col + (255,), dtype=np.float32) / 255.0
        #print(f"({pos[0]:4d},{pos[1]:4d},{pos[2]:4d}) -> ({calib(pos)[0]:4d},{calib(pos)[1]:4d},{calib(pos)[2]:4d})")

    ax.voxels(
        arr,
        facecolors=col_arr,
        alpha=0.6)

    plt.show()