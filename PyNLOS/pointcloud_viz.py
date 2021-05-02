import open3d as o3d
import scipy.io
import numpy as np
import cv2
from functools import partial


threshold = 2.6e6
voxels = None
clrs = None
maxvals = None
maxidxs = None
pc = None
oct = None
oct_max_depth = 0
mult_factor = 10.0


def main():
    global threshold
    global voxels
    global clrs
    global maxvals
    global maxidxs
    global pc
    global oct

    matW = scipy.io.loadmat('../DataCode_Phasor_Field_VWNLOS/Datasets/officescene_W.mat')
    W = np.array(matW['W'])

    # apply colormap
    maxvals = np.max(W, axis=2)
    clrs = (maxvals / np.max(maxvals) * 255.0).astype(np.uint8)
    clrs = cv2.applyColorMap(clrs, 11).astype(np.float) / 255.0


    ix,iy,iz = np.indices((W.shape[0]+1, W.shape[1]+1,W.shape[2]+1))
    voxels = np.full_like(W, False, dtype=np.bool)
    colors = np.zeros(voxels.shape + (3,), dtype=np.float)   

    geom = []
    maxidxs = np.argmax(W, axis=2)

    pc = build_pointcloud()

    geom.append(pc)

    coord_sys = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
    geom.append(coord_sys)
    geom.append(wireframe_box(size=[W.shape[0], W.shape[1], -W.shape[2]]))
    print(f'Rendering {len(geom)-2} voxels...')

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_action_callback(ord("T"), adjust_threshold)
    vis.register_key_action_callback(ord("S"), adjust_mult_factor)
    vis.register_key_action_callback(ord("O"), toggle_octree)
    vis.register_key_action_callback(ord("D"), adjust_octree_depth)
    vis.create_window(window_name='Officescene')
    for g in geom:
        vis.add_geometry(g)

    while vis.poll_events():
        vis.update_renderer()

def toggle_octree(vis, action, modifier):
    global oct
    global pc

    if action != 0:
        return True

    view_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    if oct is None:
        oct = build_octree(pc)
        vis.add_geometry(oct)
    else:
        vis.remove_geometry(oct)
        oct = None
    vis.get_view_control().convert_from_pinhole_camera_parameters(view_params)

def adjust_threshold(vis, action, modifier):
    global threshold
    global pc
    global oct
    global mult_factor

    if action != 0:
        return True

    if modifier & 1 == 1: # increase
        if threshold < mult_factor:
            threshold = mult_factor
        else:
            threshold *= mult_factor
    else: # decrease:
        if threshold > mult_factor:
            threshold /= mult_factor
        else:
            threshold  = 0

    view_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    if oct is not None:
        vis.remove_geometry(oct)
    
    vis.remove_geometry(pc)
    pc = build_pointcloud()

    if oct is not None:
        oct = build_octree(pc)
        vis.add_geometry(oct)

    vis.add_geometry(pc)
    vis.get_view_control().convert_from_pinhole_camera_parameters(view_params)

    print(f"threshold is now {threshold:.2f}")
    
    return True

def adjust_mult_factor(vis, action, modifier):
    global mult_factor

    if action != 0:
        return True

    if modifier & 1 == 1: # increase
        if mult_factor < 3:
            mult_factor = (mult_factor - 1) * 2 + 1
        else:
            mult_factor *= 2
    else: # decrease:
        if mult_factor > 3:
            mult_factor /= 2
        else:
            mult_factor = (mult_factor + 1) / 2

    print(f"threshold mult factor is now {mult_factor:.2f}")
    
    return True

def adjust_octree_depth(vis, action, modifier):
    global pc
    global oct
    global oct_max_depth

    if action != 0:
        return True

    if modifier & 1 == 1: # increase
        oct_max_depth = min(oct_max_depth+1, 10)
    else: # decrease:
        oct_max_depth = max(oct_max_depth-1, 0)

    view_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    print(f"octree max depth is now {oct_max_depth}")
    if oct is not None:
        vis.remove_geometry(oct)
        oct = build_octree(pc)
        vis.add_geometry(oct)

    vis.get_view_control().convert_from_pinhole_camera_parameters(view_params)

    return True

def build_octree(pc):
    global oct_max_depth

    oct = o3d.geometry.Octree(max_depth=oct_max_depth)
    oct.convert_from_point_cloud(pc)
    
    info = OctreeInfo(oct, pc)
    print(info)

    return oct

def build_pointcloud():
    global threshold
    global voxels
    global clrs
    global maxvals
    global maxidxs
    global pc

    points = []
    colors = []
    for x in range(voxels.shape[0]):
       for y in range(voxels.shape[1]):
           if maxvals[x,y] > threshold:
               points.append([y, 250-x, -maxidxs[x,y]])
               colors.append(clrs[x,y,:])

    print(f'Number of voxels: {len(points)}')
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc

def wireframe_box(size=[1,1,1], origin=[0,0,0], color=[1,0,0]):
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    tfm = np.identity(4)
    for i in range(3):
       tfm[i,i] = size[i]
    tfm[0:3,3] = origin
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    ).transform(tfm).paint_uniform_color(color)

    return line_set

class OctreeInfo():
    def __init__(self, oct: o3d.geometry.Octree, pc: o3d.geometry.PointCloud):
        self.oct = oct
        self.pc = pc
        self.max_depth = 0
        self.num_leafs = 0
        self.num_visited = 0
        self.num_empty_internal = 0
        self.num_internal = 0
        self._traverse(oct.root_node, 0)

    def __str__(self):
        ret = f'Octree Info:\n'
        ret += f' max_depth         : {self.max_depth}\n'
        ret += f' num_leafs         : {self.num_leafs}\n'
        ret += f' num_internal      : {self.num_internal}\n'
        ret += f' num_empty         : {self.num_empty_internal}\n'
        if self.pc is not None and not self.pc.is_empty():
            ret += f' all in leaf       : {self.num_leafs == len(self.pc.points)}\n'
        ret += f' num_reconstructed : {self.num_visited}\n'
        ret += f' % reconstructed   : {self.num_visited / (250*250*200) * 100:.2f}% ({self.num_visited}/{250*250*200})\n'
        return ret
    
    def _traverse(self, node, cur_depth):
        self.max_depth = max(cur_depth, self.max_depth)

        if isinstance(node, o3d.geometry.OctreeInternalNode):
            self.num_internal += 1
            self.num_visited += 1
            for child in node.children:
                if child is None:
                    self.num_empty_internal += 1
                    self.num_visited += 1
                else:
                    self._traverse(child, cur_depth+1)
        elif isinstance(node, o3d.geometry.OctreeColorLeafNode):
            self.num_leafs += 1
            self.num_visited += 1

if __name__ == '__main__':
    main()