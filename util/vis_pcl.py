import numpy as np
import pyvista
from enum import Enum

bone = [[0, 1], [1, 2], [2, 3],
                [4, 5], [5, 6], [6, 7],
                [8, 9], [9, 10], [10, 11],
                [12, 13], [13, 14], [14, 15],
                [16, 17], [17, 18], [18, 19],
                [3, 20], [7, 20], [11, 20], [15, 20], [19, 20],
                [20, 21], [20, 22]]


THUMB = [0, 0, 255]
INDEX = [75, 255, 66]
MIDDLE = [255, 0, 0]
RING = [17, 240, 244]
LITTLE = [160,32,240]
WRIST = [255, 0, 255]
ROOT = [255, 0, 255]

joint_color = [LITTLE,LITTLE,LITTLE,LITTLE,
                RING,RING,RING,RING,
                MIDDLE,MIDDLE,MIDDLE,MIDDLE,
                INDEX, INDEX,INDEX, INDEX,
                THUMB,THUMB,THUMB,THUMB,
                WRIST,WRIST,WRIST]

bone_color = [LITTLE,LITTLE,LITTLE,
                RING,RING,RING,
                MIDDLE,MIDDLE,MIDDLE,
                INDEX,INDEX,INDEX,
                THUMB,THUMB,THUMB,
                LITTLE, RING, MIDDLE, INDEX, THUMB, WRIST,
                WRIST,WRIST]

def compute_vectors(points, origin):
    vectors = origin - points
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    return vectors


point_cloud = np.loadtxt('/home/cyc/pycharm/lxy/fusionTR/checkpoint/dexycb/RGBD_[visual]__Multi_doubleTR______MANO-Self-MANO-resnet-18_ips128/debug/aligned_depth_to_color_000054___pcl_53.txt')
##mesh = np.loadtxt('./debug/mesh_0.txt')

# point_cloud = np.concatenate((point_cloud, mesh), axis=0)
pdata = pyvista.PolyData(point_cloud)
# pdata['orig_sphere'] = (point_cloud[:, 2] + 1)/2
# pdata['point_color'] = (point_cloud[:, 2] + 1)/2

joint_index = 16
##joint = np.loadtxt('./debug/joint_0.txt')[joint_index:joint_index+1, :]
# joint += [0, 0, 0.04]
##jdata = pyvista.PolyData(joint)

##mdata = pyvista.PolyData(mesh)

pl = pyvista.Plotter()

# 直接利用点云绘制
pyvista.plot(pdata, render_points_as_spheres=True, background='white', cmap='jet')

pdata.save('/home/cyc/pycharm/lxy/fusionTR/checkpoint/dexycb/RGBD_[visual]__Multi_doubleTR______MANO-Self-MANO-resnet-18_ips128/debug/test.pdf')


axes = pyvista.Axes(show_actor=True, actor_scale=1.0, line_width=1)
axes.origin = (0.0, 0.0, 0.0)
# pl.add_actor(axes.actor)

##mdata = mdata.rotate_x(-110, point=axes.origin,inplace=True)
##mdata = mdata.rotate_z(110, point=axes.origin,inplace=True)
##mdata = mdata.rotate_x(20, point=axes.origin,inplace=True)

pdata = pdata.rotate_x(-110, point=axes.origin,inplace=True)
pdata = pdata.rotate_z(110, point=axes.origin,inplace=True)
pdata = pdata.rotate_x(20, point=axes.origin,inplace=True)


##jdata = jdata.rotate_x(-110, point=axes.origin,inplace=True)
##jdata = jdata.rotate_z(110, point=axes.origin,inplace=True)
##jdata = jdata.rotate_x(20, point=axes.origin,inplace=True)


# 显式原始点云
sphere = pyvista.Sphere(radius=0.01, phi_resolution=50, theta_resolution=50)
pc = pdata.glyph(scale=False, geom=sphere, orient=False)
pl.add_mesh(pc, color='gray', smooth_shading=False)


# 计算距离
# dis = np.sqrt(np.sum((joint - point_cloud)**2, axis=-1))
# dis = np.clip(0.15-dis, 0, 0.15)
# dis = np.clip((dis + 0.1), 0, 0.15)
# # 添加距离
# pdata['point_color'] = dis
# sphere = pyvista.Sphere(radius=0.01, phi_resolution=50, theta_resolution=50)
# pc = pdata.glyph(scale=False, geom=sphere, orient=False)
# pl.add_mesh(pc, cmap='jet', smooth_shading=False)

# 添加方向向量
# vector_cloud = pdata.points[np.where(dis>0)]
# print(vector_cloud.shape)
# vdata = pyvista.PolyData(vector_cloud)
# vdata['vectors'] = compute_vectors(vector_cloud, jdata.points)
# arrows = vdata.glyph(orient='vectors', scale=False, factor=0.06)
# pl.add_mesh(arrows, color='red')

# sphere = pyvista.Sphere(radius=0.02, phi_resolution=50, theta_resolution=50)
# node_data = pyvista.PolyData(jdata.points)
# node_data = node_data.glyph(scale=False, geom=sphere, orient=False)
# pl.add_mesh(node_data, smooth_shading=True, color=joint_color[joint_index])

# sphere = pyvista.Sphere(radius=0.015, phi_resolution=50, theta_resolution=50)
# mc = pdata.glyph(scale=False, geom=sphere, orient=False)
# pl.add_mesh(mc, color='gray', smooth_shading=True)

# 添加所有关节点
# for index in range(23):
#     sphere = pyvista.Sphere(radius=0.02, phi_resolution=50, theta_resolution=50)
#     node_data = pyvista.PolyData(jdata.points[index])
#     node_data = node_data.glyph(scale=False, geom=sphere, orient=False)
#     pl.add_mesh(node_data, smooth_shading=True, color=joint_color[index])
#
# # 添加骨骼
# for index, bone_id in enumerate(bone):
#     mesh = pyvista.Line(jdata.points[bone_id[0]], jdata.points[bone_id[1]])
#     pl.add_mesh(mesh, color=bone_color[index], line_width=5)

# pl.show_bounds(padding=0.1
#                 ,color='black',
#                grid='front',
#                 location='outer',
#                 all_edges=False,
#                xlabel='', ylabel='', zlabel='')
pl.set_background('white')
# pl.view_isometric()
pl.save_graphic('/home/cyc/pycharm/lxy/fusionTR/checkpoint/dexycb/RGBD_[visual]__Multi_doubleTR______MANO-Self-MANO-resnet-18_ips128/debug/test.svg')
#pl.show()

