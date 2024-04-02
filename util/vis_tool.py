import torch
import cv2
from enum import Enum
from matplotlib import cm
import matplotlib.colors as colors
# import matplotlib.pyplot as plt
import numpy as np


def get_param(dataset):
    if dataset == 'icvl' or dataset == 'nyu':
        return 240.99, 240.96, 160, 120
    elif dataset == 'msra':
        return 241.42, 241.42, 160, 120
    elif dataset == 'FHAD' or 'hands' in dataset:
        return 475.065948, 475.065857, 315.944855, 245.287079
    elif dataset == 'itop':
        return 285.71, 285.71, 160.0, 120.0


def get_joint_num(dataset):
    if dataset == 'nyu':
        return 14
    elif dataset == 'icvl':
        return 16
    elif dataset == 'FHAD' or 'hands' in dataset or 'msra' in dataset:
        return 21
    elif dataset == 'itop':
        return 15


def pixel2world(x, dataset):
    fx, fy, ux, uy = get_param(dataset)
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


def world2pixel(x, dataset):
    fx,fy,ux,uy = get_param(dataset)
    x[:, :, 0] = x[:, :, 0] * fx/x[:, :, 2] + ux
    x[:, :, 1] = uy - x[:, :, 1] * fy / x[:, :, 2]
    return x


def jointImgTo3D(uvd, paras):
    fx, fy, fu, fv = paras
    ret = np.zeros_like(uvd, np.float32)
    if len(ret.shape) == 1:
        ret[0] = (uvd[0] - fu) * uvd[2] / fx
        ret[1] = (uvd[1] - fv) * uvd[2] / fy
        ret[2] = uvd[2]
    else:
        ret[:, 0] = (uvd[:,0] - fu) * uvd[:, 2] / fx
        ret[:, 1] = (uvd[:,1] - fv) * uvd[:, 2] / fy
        ret[:, 2] = uvd[:,2]
    return ret


def joint3DToImg(xyz, paras):
    fx, fy, fu, fv = paras
    ret = np.zeros_like(xyz, np.float32)
    if len(ret.shape) == 1:
        ret[0] = (xyz[0] * fx / xyz[2] + fu)
        ret[1] = (xyz[1] * fy / xyz[2] + fv)
        ret[2] = xyz[2]
    else:
        ret[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
        ret[:, 1] = (xyz[:, 1] * fy / xyz[:, 2] + fv)
        ret[:, 2] = xyz[:, 2]
    return ret


def get_sketch_setting(dataset):
    if dataset == 'FHAD' or 'hands' in dataset:
        return [
                [0, 13], [13, 14], [14, 15], [15, 20],
                [0, 1], [1, 2], [2, 3], [3, 16],
                [0, 4], [4, 5], [5, 6], [6, 17],
                [0, 10], [10, 11], [11,  12], [12, 19],
                [0, 7], [7, 8], [8, 9], [9, 18]
                ]
        # return [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
        #         [1, 6], [6, 7], [7, 8],
        #         [2, 9], [9, 10], [10, 11],
        #         [3, 12], [12, 13],[13, 14],
        #         [4, 15], [15, 16],[16, 17],
        #         [5, 18], [18, 19], [19, 20]]
    elif 'nyu' == dataset:
        return [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [9, 10], [1, 13],
                [3, 13], [5, 13], [7, 13], [10, 13], [11, 13], [12, 13]]
    elif 'nyu_all' == dataset:
        return [[0, 1], [1, 2], [2, 3],
                [4, 5], [5, 6], [6, 7],
                [8, 9], [9, 10], [10, 11],
                [12, 13], [13, 14], [14, 15],
                [16, 17], [17, 18], [18, 19],
                [3, 20], [7, 20], [11, 20], [15, 20], [19, 20],
                [20, 21], [20, 22]]
    elif dataset == 'icvl':
        return [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
                [0, 7], [7, 8], [8, 9], [0, 10], [10, 11], [11, 12],
                [0, 13], [13, 14], [14, 15]]
    elif dataset == 'msra':
        return [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
                [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
                [0, 17], [17, 18], [18, 19], [19, 20]]
    elif dataset == 'itop':
        return [[0, 1],
                [1, 2], [2, 4], [4, 6],
                [1, 3], [3, 5], [5, 7],
                [1, 8],
                [8, 9], [9, 11], [11, 13],
                [8, 10], [10, 12], [12, 14]]
    elif dataset == 'shrec' or 'DHG' in dataset:
        return [[0, 1],
                [0, 2], [2, 3], [3, 4], [4, 5],
                [0, 6], [6, 7], [7, 8], [8, 9],
                [0, 10], [10, 11], [11, 12], [12, 13],
                [0, 14], [14, 15], [15, 16], [16, 17],
                [0, 18], [18, 19], [19, 20], [20 ,21]]
    else:
        return [
                [0, 13], [13, 14], [14, 15], [15, 20],
                [0, 1], [1, 2], [2, 3], [3, 16],
                [0, 4], [4, 5], [5, 6], [6, 17],
                [0, 10], [10, 11], [11,  12], [12, 19],
                [0, 7], [7, 8], [8, 9], [9, 18]
                ]


def get_hierarchy_mapping(dataset):
    if 'mano' in dataset or 'hands' in dataset:
        return [[[0], [1, 2], [3, 16], [4, 5], [6,17], [10, 11], [12, 19], [7, 8],[9, 18], [13, 14],[15,20]],\
            [[0], [1, 2], [3, 4], [7, 8], [5, 6], [9, 10]], \
            [[0, 1, 2, 3, 4, 5]],
                ]
    elif 'nyu' == dataset:
        return [[[0, 1], [2,3], [4,5], [6,7], [8,9,10], [11,12,13]], ]
    elif 'nyu_all' == dataset:
        return [[[0, 1], [2, 3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15], [16,17],[18,19],[20]],\
                [[0,1], [2,3], [4,5], [6,7], [8,9], [10]],\
                [[0, 1, 2, 3, 4, 5]]]

def debug_mesh(verts, faces, batch_index, data_dir, img_type):
    batch_size = verts.size(0)
    verts = verts.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()
    for index in range(batch_size):
        path = data_dir + '/' + str(batch_index * batch_size + index) + '_' + img_type + '.obj'
        with open(path, 'w') as fp:
            for v in verts[index]:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

def get_hierarchy_sketch(dataset):
    if 'nyu' == dataset:
        return [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [9, 10], [1, 13],
                [3, 13], [5, 13], [7, 13], [10, 13], [11, 13], [12, 13]], \
               [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [1, 5], [2, 5], [3, 5], [4, 5]]
    elif 'nyu_all' == dataset:
        return [[0, 1], [1, 2], [2, 3],
                [4, 5], [5, 6], [6, 7],
                [8, 9], [9, 10], [10, 11],
                [12, 13], [13, 14], [14, 15],
                [16, 17], [17, 18], [18, 19],
                [3, 20], [7, 20], [11, 20], [15, 20], [19, 20],[20,21],[20,22]],\
               [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9],[1,10],[3,10],[5,10],[7,10],[9,10]],\
               [[0, 5], [1, 5], [2, 5], [3, 5], [4, 5]], \
               [[0, 0]]
    elif 'mano' == dataset or 'hands' in dataset:
        return [
                [0, 13], [13, 14], [14, 15], [15, 20],
                [0, 1], [1, 2], [2, 3], [3, 16],
                [0, 4], [4, 5], [5, 6], [6, 17],
                [0, 10], [10, 11], [11,  12], [12, 19],
                [0, 7], [7, 8], [8, 9], [9, 18]
                ],\
                [[0, 1], [0, 3], [0, 5], [0, 7], [0, 9], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], \
               [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],\
                [[0, 0]]


class Color(Enum):
    RED = (0, 0, 255)
    GREEN = (75, 255, 66)
    BLUE = (255, 0, 0)
    YELLOW = (204, 153, 17) #(17, 240, 244)
    PURPLE = (255, 255, 0)
    CYAN = (255, 0, 255)
    BROWN = (204, 153, 17)


class Finger_color(Enum):
    THUMB = (0, 0, 255)
    INDEX = (75, 255, 66)
    MIDDLE = (255, 0, 0)
    RING = (17, 240, 244)
    LITTLE = (255, 255, 0)
    WRIST = (255, 0, 255)
    ROOT = (255, 0, 255)


def get_sketch_color(dataset):
    if dataset == 'FHAD' or 'hands' in dataset:
        return (Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
                Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
               Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
                Finger_color.RING, Finger_color.RING, Finger_color.RING, Finger_color.RING,
               Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE)
        # return [Finger_color.THUMB, Finger_color.INDEX, Finger_color.MIDDLE, Finger_color.RING, Finger_color.LITTLE,
        #         Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
        #         Finger_color.INDEX,  Finger_color.INDEX,  Finger_color.INDEX,
        #       Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
        #       Finger_color.RING, Finger_color.RING, Finger_color.RING,
        #       Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE,
        #       ]
    elif dataset == 'nyu':
        return (Finger_color.LITTLE,Finger_color.RING,Finger_color.MIDDLE,Finger_color.INDEX,Finger_color.THUMB,Finger_color.THUMB,
                Finger_color.LITTLE, Finger_color.RING, Finger_color.MIDDLE, Finger_color.INDEX, Finger_color.THUMB, Finger_color.THUMB,
                Finger_color.WRIST,Finger_color.WRIST)
    elif dataset == 'nyu_all':
        return (Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE,
                Finger_color.RING,Finger_color.RING,Finger_color.RING,
                Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE,
                Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,
                Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,
                Finger_color.LITTLE, Finger_color.RING, Finger_color.MIDDLE, Finger_color.INDEX, Finger_color.THUMB, Finger_color.THUMB,
                Finger_color.WRIST,Finger_color.WRIST)
    elif dataset == 'icvl':
        return [Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,
                Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE, Finger_color.RING,Finger_color.RING,Finger_color.RING,
                Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE]
    elif dataset == 'msra':
        return [Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,
                 Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE,
                 Finger_color.RING,Finger_color.RING,Finger_color.RING,Finger_color.RING,
                 Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE,
                 Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB]
    elif dataset == 'itop':
        return [Color.RED,
                Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE,
                Color.CYAN,
                Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE,
                ]
    elif dataset == 'shrec' or 'DHG' in dataset:
        return (Finger_color.ROOT,
            Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
         Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
         Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
         Finger_color.RING, Finger_color.RING, Finger_color.RING, Finger_color.RING,
         Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE,)
    else:
        return (Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
                Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
               Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
                Finger_color.RING, Finger_color.RING, Finger_color.RING, Finger_color.RING,
               Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE)


def get_joint_color(dataset):
    if dataset == 'FHAD'or 'hands' in dataset:
        return [Finger_color.ROOT,
                 Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
                 Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
                 Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE,
                 Finger_color.RING, Finger_color.RING, Finger_color.RING,
                 Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
                 Finger_color.INDEX, Finger_color.MIDDLE, Finger_color.LITTLE, Finger_color.RING, Finger_color.THUMB,
                ]
        # return [Finger_color.ROOT,
        #         Finger_color.THUMB, Finger_color.INDEX, Finger_color.MIDDLE, Finger_color.RING, Finger_color.LITTLE,
        #         Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
        #         Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
        #         Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
        #         Finger_color.RING, Finger_color.RING, Finger_color.RING,
        #         Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE]
    elif dataset == 'nyu':
        return [Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.RING,Finger_color.RING,Finger_color.MIDDLE,Finger_color.MIDDLE,
                Finger_color.INDEX, Finger_color.INDEX,Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,
                Finger_color.WRIST,Finger_color.WRIST,Finger_color.WRIST]
    elif dataset == 'nyu_all':
        return [Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE,
                Finger_color.RING,Finger_color.RING,Finger_color.RING,Finger_color.RING,
                Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE,
                Finger_color.INDEX, Finger_color.INDEX,Finger_color.INDEX, Finger_color.INDEX,
                Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,
                Finger_color.WRIST,Finger_color.WRIST,Finger_color.WRIST]
    if dataset == 'icvl':
        return [Finger_color.ROOT,Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,
                 Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,
                 Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE,
                 Finger_color.RING,Finger_color.RING,Finger_color.RING,
                 Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE]
    elif dataset == 'msra':
        return [Finger_color.WRIST,Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,Finger_color.INDEX,Finger_color.MIDDLE,
                Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.MIDDLE,Finger_color.RING,Finger_color.RING,Finger_color.RING,Finger_color.RING,
                Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.LITTLE,Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB,Finger_color.THUMB]
    elif dataset == 'itop':
        return  [Color.RED,Color.BROWN,
                 Color.GREEN, Color.BLUE, Color.GREEN, Color.BLUE, Color.GREEN, Color.BLUE,
                 Color.CYAN,
                 Color.YELLOW,Color.PURPLE,Color.YELLOW,Color.PURPLE,Color.YELLOW,Color.PURPLE]
    elif dataset == 'shrec' or 'DHG' in dataset:
        return [Finger_color.ROOT, Finger_color.ROOT,
            Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
         Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
         Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
         Finger_color.RING, Finger_color.RING, Finger_color.RING, Finger_color.RING,
         Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE,]
    else:
        return [Finger_color.ROOT,
                 Finger_color.INDEX, Finger_color.INDEX, Finger_color.INDEX,
                 Finger_color.MIDDLE, Finger_color.MIDDLE, Finger_color.MIDDLE,
                 Finger_color.LITTLE, Finger_color.LITTLE, Finger_color.LITTLE,
                 Finger_color.RING, Finger_color.RING, Finger_color.RING,
                 Finger_color.THUMB, Finger_color.THUMB, Finger_color.THUMB,
                 Finger_color.INDEX, Finger_color.MIDDLE, Finger_color.LITTLE, Finger_color.RING, Finger_color.THUMB,
                ]


def draw_point(dataset, img, pose):
    colors_joint = get_joint_color(dataset)
    idx = 0
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, colors_joint[0].value, -1)
        idx = idx + 1
    return img


def draw_pose(dataset, img, pose, scale=1):

    colors_joint = get_joint_color(dataset)
    idx = 0
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 2*scale, colors_joint[idx].value, -1)
        idx = idx + 1
        if idx >= len(colors_joint):
            break
    colors = get_sketch_color(dataset)
    idx = 0
    for index, (x, y) in enumerate(get_sketch_setting(dataset)):
        if x >= pose.shape[0] or y >= pose.shape[0]:
            break
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), colors[idx].value, 1*scale)
        idx = idx + 1
    return img

import torch.nn.functional as F
def debug_img_heatmap(img, heatmap2d, batch_index, data_dir, size, img_type='heatmap', save=False):
    cNorm = colors.Normalize(vmin=0, vmax=1.0)
    jet = plt.get_cmap('jet')
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    batch_size, head_num, height, width = heatmap2d.size()
    heatmap2d = heatmap2d.view(batch_size,head_num,-1)
    heatmap2d = (heatmap2d - heatmap2d.min(dim=-1, keepdim=True)[0])
    heatmap2d = heatmap2d / (heatmap2d.max(dim=-1, keepdim=True)[0] + 1e-8)
    heatmap2d = heatmap2d.view(batch_size, head_num, height, width)
    img = F.interpolate(img, (height, width))
    heatmap_list = []
    heatmap = heatmap2d.cpu().detach().numpy()
    img = (img.cpu().detach().numpy()+1)/2*255
    for index in range(heatmap2d.size(0)):
        for joint_index in range(heatmap2d.size(1)):
                img_dir = data_dir + '/' + img_type + '_' + str(batch_size * batch_index + index) + '_' + \
                          str(joint_index) + '.png'
                heatmap_draw = cv2.resize(heatmap[index, joint_index], (size, size))
                heatmap_color = 255 * scalarMap.to_rgba(1 - heatmap_draw)
                img_draw = cv2.cvtColor(img[index, 0], cv2.COLOR_GRAY2RGB)/2 + heatmap_color.reshape(size, size, 4)[:, :, 0:3]
                if save:
                    cv2.imwrite(img_dir, img_draw)
                heatmap_list.append(img_draw)
    return np.stack(heatmap_list, axis=0).squeeze()


def debug_2d_heatmap(heatmap2d, batch_index, data_dir, size, img_type='heatmap', save=False):
    cNorm = colors.Normalize(vmin=0, vmax=1.0)
    jet = plt.get_cmap('jet')
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    batch_size, head_num, height, width = heatmap2d.size()
    if batch_size==0:
        return 0
    # heatmap2d = heatmap2d.view(batch_size,head_num,-1)
    # heatmap2d = (heatmap2d - heatmap2d.min(dim=-1, keepdim=True)[0])
    # heatmap2d = heatmap2d / (heatmap2d.max(dim=-1, keepdim=True)[0] + 1e-8)
    # heatmap2d = heatmap2d.view(batch_size, head_num, height, width)

    # heatmap2d = F.interpolate(heatmap2d, size=[128, 128])
    heatmap_list = []
    heatmap = heatmap2d.cpu().detach().numpy()
    for index in range(heatmap2d.size(0)):
        for joint_index in range(heatmap2d.size(1)):
                img_dir = data_dir + '/' + img_type + '_' + str(batch_size * batch_index + index) + '_' + str(
                    joint_index) + '.png'
                heatmap_draw = cv2.resize(heatmap[index, joint_index], (size, size))
                heatmap_color = 255 * scalarMap.to_rgba(1 - heatmap_draw)
                if save:
                    cv2.imwrite(img_dir, heatmap_color.reshape(size, size, 4)[:, :, 0:3])
                heatmap_list.append(heatmap_color.reshape(size, size, 4)[:, :, 0:3])
                # ret, img_show = cv2.threshold(img_draw[index, 0] * 255.0, 245, 255, cv2.THRESH_BINARY)
                # img_show = cv2.cvtColor(img_show, cv2.COLOR_GRAY2RGB)
                # cv2.imwrite(img_dir, img_show/2 + heatmap_color.reshape(128, 128, 4)[:, :, 0:3])
    return np.stack(heatmap_list, axis=0).squeeze()


def debug_offset(data, batch_index, GFM_):
    img, pcl_sample, joint_world, joint_img, center, M, cube, pcl_normal, joint_normal, offset, coeff, max_bbx_len = data
    img_size = 32
    batch_size,joint_num,_ = joint_world.size()
    offset = GFM_.joint2offset(joint_img, img, feature_size=img_size)
    unit = offset[:, 0:joint_num*3, :, :].numpy()
    for index in range(batch_size):
        fig, ax = plt.subplots()
        unit_plam = unit[index, 0:3, :, :]
        x = np.arange(0,img_size,1)
        y = np.arange(0,img_size,1)

        X, Y = np.meshgrid(x, y)
        Y = img_size - 1 - Y
        ax.quiver(X, Y, unit_plam[0, ...], unit_plam[1, ...])
        ax.axis([0, img_size, 0, img_size])
        plt.savefig('./debug/offset_' + str(batch_index) + '_' + str(index) + '.png')


def debug_offset_heatmap(img, joint, batch_index, GFM_, kernel_size):
    img_size = 128
    batch_size,joint_num,_ = joint.size()
    offset = GFM_.joint2offset(joint, img, kernel_size, feature_size=img_size)
    heatmap = offset[:, joint_num*3:, :, :].numpy()
    cNorm = colors.Normalize(vmin=0, vmax=1.0)
    jet = plt.get_cmap('jet')
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    img_draw = img.numpy()
    for index in range(batch_size):
        for joint_index in range(joint_num):
            img_dir = './debug/' + str(batch_size * batch_index + index) + '_' + str(joint_index) + '.png'
            heatmap_color = 255 * scalarMap.to_rgba((kernel_size-heatmap[index, joint_index].reshape(128, 128)) / kernel_size)
            img_show = cv2.cvtColor(img_draw[index, 0] * 255.0/2.0, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(img_dir, img_show + heatmap_color.reshape(128, 128, 4)[:, :, 0:3])


def debug_2d_img(img, index, data_dir, name, batch_size):
    _, num, input_size, input_size = img.size()
    img_list = []
    for img_idx in range(img.size(0)):
        for channel_idx in range(img.size(1)):
            img_draw = (img.detach().cpu().numpy()[img_idx,channel_idx] + 1) / 2 * 255
            img_draw = cv2.cvtColor(img_draw, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(data_dir + '/' + str(batch_size * index + img_idx) + '_'+str(channel_idx)+"_" + name + '.png', img_draw)
            img_list.append(img_draw)
    return np.stack(img_list, axis=0)


def debug_2d_pose(img, joint_img, index, dataset, data_dir, name, batch_size, save=False):
    _, num, input_size, input_size = img.size()
    img_list = []
    for img_idx in range(joint_img.size(0)):
        joint_uvd = (joint_img.detach().cpu().numpy() + 1) / 2 * input_size
        img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255
        img_show = draw_pose(dataset, cv2.cvtColor(img_draw[img_idx, 0], cv2.COLOR_GRAY2RGB),
                             joint_uvd[img_idx], input_size // 128)
        if save:
           cv2.imwrite(data_dir + '/' + str(batch_size * index + img_idx) + '_' + name + '.png', img_show)
        img_list.append(img_show)
    return np.stack(img_list, axis=0)



def debug_2d_pose_select(img, joint_img, index, dataset, data_dir, name, batch_size, select_id, save=False):
    _, num, input_size, input_size = img.size()
    img_list = []
    for img_index, img_id in enumerate(select_id):
        joint_uvd = (joint_img.detach().cpu().numpy() + 1) / 2 * input_size
        img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255
        img_show = draw_pose(dataset, cv2.cvtColor(img_draw[img_index, 0], cv2.COLOR_GRAY2RGB),
                             joint_uvd[img_index], input_size // 128)
        if save:
           cv2.imwrite(data_dir + '/' + str(batch_size * index + img_id) + '_' + name + '.png', img_show)
        img_list.append(img_show)
    # return np.stack(img_list, axis=0)
    return 0

def draw_2d_pose(img, joint_img, dataset):
    num, input_size, input_size = img.size()
    joint_uvd = (joint_img.detach().cpu().numpy() + 1) / 2 * input_size
    img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255
    img_show = draw_pose(dataset, cv2.cvtColor(img_draw[0], cv2.COLOR_GRAY2RGB), joint_uvd)
    return img_show

def draw_2d_pose_only(img, joint_img, dataset):
    num, input_size, input_size = img.size()
    img[:,:,:] = 1
    joint_uvd = (joint_img.detach().cpu().numpy() + 1) / 2 * input_size
    img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255
    img_show = draw_pose(dataset, cv2.cvtColor(img_draw[0], cv2.COLOR_GRAY2RGB), joint_uvd)
    return img_show

def draw_2d_pose_rgb(img, joint_img, dataset):
    num, input_size, input_size = img.size()
    joint_uvd = (joint_img.detach().cpu().numpy() + 1) / 2 * input_size
    img_draw = img.detach().permute(1, 2, 0).cpu().numpy() * 255
    np.uint8(img_draw)
    #print(img_draw.shape)
    img_show = draw_pose(dataset, img_draw.copy(), joint_uvd)

    return img_show


def draw_visible(dataset, img, pose, visible):
    idx = 0
    color = [Color.RED, Color.BLUE]
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, color[visible[idx]].value, -1)
        idx = idx + 1
    idx = 0
    for x, y in get_sketch_setting(dataset):
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), Color.BROWN.value, 1)
        idx = idx + 1
    return img


def debug_visible_joint(img, joint_img, visible, index, dataset, data_dir, name):
    batch_size,_,input_size,input_size = img.size()
    visible = visible.detach().cpu().numpy().astype(np.int)
    for img_idx in range(img.size(0)):
        joint_uvd = (joint_img.detach().cpu().numpy() + 1) / 2 * input_size
        img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255
        img_show = draw_visible(dataset, cv2.cvtColor(img_draw[img_idx, 0], cv2.COLOR_GRAY2RGB), joint_uvd[img_idx], visible[img_idx])
        cv2.imwrite(data_dir + '/' + str(batch_size * index + img_idx) + '_' + name + '.png', img_show)


def draw_pcl(pcl, img_size, background_value=1):
    device = pcl.device
    batch_size = pcl.size(0)
    img_pcl = []
    for index in range(batch_size):
        img = torch.ones([img_size, img_size]).to(device) * background_value
        index_x = torch.clamp(torch.floor((pcl[index, :, 0] + 1) / 2 * img_size), 0, img_size - 1).long()
        index_y = torch.clamp(torch.floor((pcl[index, :, 1] + 1) / 2 * img_size), 0, img_size - 1).long()
        img[index_y, index_x] = -1
        img_pcl.append(img)
    return torch.stack(img_pcl, dim=0).unsqueeze(1)


def debug_pcl_pose(pcl, joint_xyz, index, dataset, data_dir, name):
    """
    :param pcl:
    :param joint_xyz:
    :param index:
    :param dataset:
    :param data_dir:
    :param name:
    :return:
    """
    batch_size = pcl.size(0)
    if batch_size == 0:
        return 0
    img = draw_pcl(pcl, 128)
    for img_idx in range(img.size(0)):
        joint_uvd = (joint_xyz.detach().cpu().numpy() + 1) / 2 * 128
        img_draw = (img.detach().cpu().numpy() + 1) / 2 * 255
        im_color = cv2.cvtColor(img_draw[img_idx, 0], cv2.COLOR_GRAY2RGB)
        img_show = draw_pose(dataset, im_color, joint_uvd[img_idx])
        cv2.imwrite(data_dir + '/' + str(batch_size * index + img_idx) + '-' + name + '.png', img_show)


def draw_muti_pic(batch_img_list, index, data_dir, name, text=None, save=True, max_col=7, batch_size=32):
    # batch_size = batch_img_list[0].shape[0]
    for batch_index in range(batch_size):
        img_list = []
        img_list_temp = []
        for img_index, imgs in enumerate(batch_img_list):
            img_list_temp.append(imgs[batch_index].squeeze())
            if (img_index + 1) % max_col == 0:
                img_list.append(np.hstack(img_list_temp))
                img_list_temp = []

        if img_index < max_col:
            imgs = np.hstack(img_list_temp)
        else:
            imgs = np.concatenate(img_list, axis=0)

        if text:
            cv2.putText(imgs, text[batch_index], (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200), 1)
        if save:
            cv2.imwrite(data_dir + '/' + name + '_' + str(batch_size * index + batch_index)  + '.png', imgs)
    return imgs
