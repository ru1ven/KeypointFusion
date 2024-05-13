# -*- coding: utf-8 -*-
import os
import pickle

import cv2
import yaml
import torch
import trimesh
import numpy as np
from PIL import Image
import os.path as osp
import copy

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# from manopth.manopth.manolayer import ManoLayer
# from render.mano_layer_ho import mask_render
from util.preprocessing import load_img
import sys
import math
import scipy.io as sio
import random

sys.path.append('..')
from util import vis_tool, object_transform
from dataloader.webuser.smpl_handpca_wrapper_HAND_only import load_model
from scipy import stats, ndimage
import torch.nn.functional as F
import torchvision.transforms as transforms

joint_select = np.array([0, 1, 3, 5,
                         6, 7, 9, 11,
                         12, 13, 15, 17,
                         18, 19, 21, 23,
                         24, 25, 27, 28,
                         32, 30, 31])
# calculate = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 21, 22, 20]
calculate = [0, 1, 2, 3,
             4, 5, 6, 7,
             8, 9, 10, 11,
             12, 13, 14, 15,
             16, 17, 18, 19,
             20, 21, 22]
HANDS2MANO = [
    0,
    2, 9, 10,
    3, 12, 13,
    5, 18, 19,
    4, 15, 16,
    1, 6, 7,
    11, 14, 20, 17, 8
]
DexYCB2MANO = [
    0,
    5, 6, 7,
    9, 10, 11,
    17, 18, 19,
    13, 14, 15,
    1, 2, 3,
    8, 12, 20, 16, 4
]
HO3D2MANO = [0,
             1, 2, 3,
             4, 5, 6,
             7, 8, 9,
             10, 11, 12,
             13, 14, 15,
             17,
             18,
             20,
             19,
             16]
MANO2HO3D = [0,
             1, 2, 3,
             4, 5, 6,
             7, 8, 9,
             10, 11, 12,
             13, 14, 15,
             20, 16, 17, 19, 18]
xrange = range

_MANO_JOINTS = [
    'wrist',
    'thumb_mcp',
    'thumb_pip',
    'thumb_dip',
    'thumb_tip',
    'index_mcp',
    'index_pip',
    'index_dip',
    'index_tip',
    'middle_mcp',
    'middle_pip',
    'middle_dip',
    'middle_tip',
    'ring_mcp',
    'ring_pip',
    'ring_dip',
    'ring_tip',
    'little_mcp',
    'little_pip',
    'little_dip',
    'little_tip'
]

_MANO_JOINT_CONNECT = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]

_BOP_EVAL_SUBSAMPLING_FACTOR = 4


def calculateCoM(dpt, minDepth, maxDepth):
    """
    Calculate the center of mass
    :param dpt: depth image
    :return: (x,y,z) center of mass
    """

    dc = dpt.copy()
    dc[dc < minDepth] = 0
    dc[dc > maxDepth] = 0
    cc = ndimage.measurements.center_of_mass(dc > 0)
    num = np.count_nonzero(dc)
    com = np.array((cc[1] * num, cc[0] * num, dc.sum()), np.float64)

    if num == 0:
        return np.array((300, 300, 500), np.float64)
    else:
        return com / num


def pixel2world(x, y, z, paras):
    fx, fy, fu, fv = paras
    worldX = (x - fu) * z / fx
    worldY = (fv - y) * z / fy
    return worldX, worldY


def pixel2world_noflip(x, y, z, paras):
    fx, fy, fu, fv = paras
    worldX = (x - fu) * z / fx
    worldY = (y - fv) * z / fy
    return worldX, worldY


def world2pixel(x, y, z, paras):
    fx, fy, fu, fv = paras
    pixelX = x * fx / z + fu
    pixelY = fv - y * fy / z
    return pixelX, pixelY


def rotatePoint2D(p1, center, angle):
    """
    Rotate a point in 2D around center
    :param p1: point in 2D (u,v,d)
    :param center: 2D center of rotation
    :param angle: angle in deg
    :return: rotated point
    """
    alpha = angle * np.pi / 180.
    pp = p1.copy()
    pp[0:2] -= center[0:2]
    pr = np.zeros_like(pp)
    pr[0] = pp[0] * np.cos(alpha) - pp[1] * np.sin(alpha)
    pr[1] = pp[0] * np.sin(alpha) + pp[1] * np.cos(alpha)
    pr[2] = pp[2]
    ps = pr
    ps[0:2] += center[0:2]
    return ps


def rotatePoints2D(pts, center, angle):
    """
    Transform points in 2D coordinates
    :param pts: point coordinates
    :param center: 2D center of rotation
    :param angle: angle in deg
    :return: rotated points
    """
    ret = pts.copy()
    for i in range(pts.shape[0]):
        ret[i] = rotatePoint2D(pts[i], center, angle)
    return ret


def transformPoint2D(pt, M):
    """
    Transform point in 2D coordinates
    :param pt: point coordinates
    :param M: transformation matrix
    :return: transformed point
    """
    pt2 = np.dot(np.asarray(M).reshape((3, 3)), np.asarray([pt[0], pt[1], 1]))
    return np.asarray([pt2[0] / pt2[2], pt2[1] / pt2[2]])


def transformPoints2D(pts, M):
    """
    Transform points in 2D coordinates
    :param pts: point coordinates
    :param M: transformation matrix
    :return: transformed points
    """
    ret = pts.copy()
    for i in range(pts.shape[0]):
        ret[i, 0:2] = transformPoint2D(pts[i, 0:2], M)
    return ret


def nyu_reader(img_path):
    img = cv2.imread(img_path)
    depth = np.asarray(img[:, :, 0] + img[:, :, 1] * 256, dtype=np.float32)
    return depth


def icvl_reader(img_path):
    img = Image.open(img_path)  # open image
    assert len(img.getbands()) == 1  # ensure depth image
    depth = np.asarray(img, np.float32)
    return depth


def msra_reader(image_name, para):
    f = open(image_name, 'rb')
    data = np.fromfile(f, dtype=np.uint32)
    width, height, left, top, right, bottom = data[:6]
    depth = np.zeros((height, width), dtype=np.float32)
    f.seek(4 * 6)
    data = np.fromfile(f, dtype=np.float32)
    depth[top:bottom, left:right] = np.reshape(data, (bottom - top, right - left))
    depth_pcl = np.reshape(data, (bottom - top, right - left))
    # convert to world
    imgHeight, imgWidth = depth_pcl.shape
    hand_3d = np.zeros([3, imgHeight * imgWidth])
    d2Output_x = np.tile(np.arange(imgWidth), (imgHeight, 1)).reshape(imgHeight, imgWidth).astype('float64') + left
    d2Output_y = np.repeat(np.arange(imgHeight), imgWidth).reshape(imgHeight, imgWidth).astype('float64') + top
    hand_3d[0], hand_3d[1] = pixel2world(d2Output_x.reshape(-1), d2Output_y.reshape(-1), depth_pcl.reshape(-1), para)
    hand_3d[2] = depth_pcl.reshape(-1)
    valid = np.arange(0, imgWidth * imgHeight)
    valid = valid[(hand_3d[0, :] != 0) | (hand_3d[1, :] != 0) | (hand_3d[2, :] != 0)]
    handpoints = hand_3d[:, valid].transpose(1, 0)

    return depth, handpoints


def hands17_reader(img_path):
    img = Image.open(img_path)  # open image
    assert len(img.getbands()) == 1  # ensure depth image
    depth = np.asarray(img, np.float32)
    return depth


from sklearn.decomposition import PCA


class loader(Dataset):
    def __init__(self, img_size=128):
        self.rng = np.random.RandomState(23455)
        self.img_size = img_size
        self.allJoints = False
        self.pca = PCA(n_components=3)
        self.sample_num = 1024
        self.flip = 1

    # numpy
    def jointImgTo3D(self, uvd, paras=None, flip=None):
        if isinstance(paras, tuple):
            fx, fy, fu, fv = paras
        else:
            fx, fy, fu, fv = self.paras
        if flip == None:
            flip = self.flip
        ret = np.zeros_like(uvd, np.float32)
        if len(ret.shape) == 1:
            ret[0] = (uvd[0] - fu) * uvd[2] / fx
            ret[1] = flip * (uvd[1] - fv) * uvd[2] / fy
            ret[2] = uvd[2]
        elif len(ret.shape) == 2:
            ret[:, 0] = (uvd[:, 0] - fu) * uvd[:, 2] / fx
            ret[:, 1] = flip * (uvd[:, 1] - fv) * uvd[:, 2] / fy
            ret[:, 2] = uvd[:, 2]
        else:
            ret[:, :, 0] = (uvd[:, :, 0] - fu) * uvd[:, :, 2] / fx
            ret[:, :, 1] = flip * (uvd[:, :, 1] - fv) * uvd[:, :, 2] / fy
            ret[:, :, 2] = uvd[:, :, 2]

        return ret

    def joint3DToImg(self, xyz, paras=None, flip=None):
        if isinstance(paras, tuple):
            fx, fy, fu, fv = paras
        else:
            fx, fy, fu, fv = self.paras
        if flip == None:
            flip = self.flip
        ret = np.zeros_like(xyz, np.float32)
        if len(ret.shape) == 1:
            ret[0] = (xyz[0] * fx / xyz[2] + fu)
            ret[1] = (flip * xyz[1] * fy / xyz[2] + fv)
            ret[2] = xyz[2]
        elif len(ret.shape) == 2:
            ret[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
            ret[:, 1] = (flip * xyz[:, 1] * fy / xyz[:, 2] + fv)
            ret[:, 2] = xyz[:, 2]
        else:
            ret[:, :, 0] = (xyz[:, :, 0] * fx / xyz[:, :, 2] + fu)
            ret[:, :, 1] = (flip * xyz[:, :, 1] * fy / xyz[:, :, 2] + fv)
            ret[:, :, 2] = xyz[:, :, 2]
        return ret

    # tensor
    def pointsImgTo3D(self, point_uvd, paras, flip=None):
        if flip == None:
            flip = self.flip
        point_xyz = torch.zeros_like(point_uvd).to(point_uvd.device)
        point_xyz[:, :, 0] = (point_uvd[:, :, 0] - paras[:, 2].unsqueeze(1)) * point_uvd[:, :, 2] / paras[:,
                                                                                                    0].unsqueeze(1)
        point_xyz[:, :, 1] = flip * (point_uvd[:, :, 1] - paras[:, 3].unsqueeze(1)) * point_uvd[:, :, 2] / paras[:,
                                                                                                           1].unsqueeze(
            1)
        point_xyz[:, :, 2] = point_uvd[:, :, 2]
        return point_xyz

    def points3DToImg(self, joint_xyz, para, flip=None):
        if flip == None:
            flip = self.flip
        joint_uvd = torch.zeros_like(joint_xyz).to(joint_xyz.device)
        joint_uvd[:, :, 0] = (
                joint_xyz[:, :, 0] * para[:, 0].unsqueeze(1) / (joint_xyz[:, :, 2] + 1e-8) + para[:, 2].unsqueeze(
            1))
        joint_uvd[:, :, 1] = (
                flip * joint_xyz[:, :, 1] * para[:, 1].unsqueeze(1) / (joint_xyz[:, :, 2]) + para[:, 3].unsqueeze(
            1))
        joint_uvd[:, :, 2] = joint_xyz[:, :, 2]
        return joint_uvd

    # augment
    def comToBounds(self, com, size, paras):
        fx, fy, fu, fv = paras
        zstart = com[2] - size[2] / 2.
        zend = com[2] + size[2] / 2.

        xstart = int(np.floor((com[0] * com[2] / fx - size[0] / 2.) / com[2] * fx + 0.5))
        xend = int(np.floor((com[0] * com[2] / fx + size[0] / 2.) / com[2] * fx + 0.5))
        ystart = int(np.floor((com[1] * com[2] / fy - size[1] / 2.) / com[2] * fy + 0.5))
        yend = int(np.floor((com[1] * com[2] / fy + size[1] / 2.) / com[2] * fy + 0.5))

        return xstart, xend, ystart, yend, zstart, zend

    def comToTransform(self, com, size, dsize, paras):
        """
        Calculate affine transform from crop
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :return: affine transform
        """

        xstart, xend, ystart, yend, _, _ = self.comToBounds(com, size, paras)

        trans = np.eye(3)
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart

        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            scale = np.eye(3) * dsize[0] / float(wb)
            sz = (dsize[0], hb * dsize[0] / wb)
        else:
            scale = np.eye(3) * dsize[1] / float(hb)
            sz = (wb * dsize[1] / hb, dsize[1])
        scale[2, 2] = 1

        # ori
        # xstart = int(np.floor(dsize[0] / 2. - sz[1] / 2.))
        # ystart = int(np.floor(dsize[1] / 2. - sz[0] / 2.))

        # change by pengfeiren
        xstart = int(np.floor(dsize[0] / 2. - sz[0] / 2.))
        ystart = int(np.floor(dsize[1] / 2. - sz[1] / 2.))
        off = np.eye(3)
        off[0, 2] = xstart
        off[1, 2] = ystart

        return np.dot(off, np.dot(scale, trans))

    def recropHand(self, crop, M, Mnew, target_size, paras, background_value=0., nv_val=0., thresh_z=True, com=None,
                   size=(250, 250, 250)):

        flags = cv2.INTER_NEAREST
        if len(target_size) > 2:
            target_size = target_size[0:2]

        warped = cv2.warpPerspective(crop, np.dot(M, Mnew), target_size, flags=flags,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=float(background_value))
        # warped[np.isclose(warped, nv_val)] = background_value # Outliers will appear on the edge
        if thresh_z:
            warped[warped < nv_val] = background_value

        if thresh_z is True:
            assert com is not None
            _, _, _, _, zstart, zend = self.comToBounds(com, size, paras)
            msk1 = np.logical_and(warped < zstart, warped != 0)
            msk2 = np.logical_and(warped > zend, warped != 0)
            warped[msk1] = zstart
            warped[msk2] = 0.  # backface is at 0, it is set later

        return warped

    def moveCoM(self, dpt, cube, com, off, joints3D, M, paras=None, pad_value=0, thresh_z=True):
        """
        Adjust already cropped image such that a moving CoM normalization is simulated
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param off: offset to center of mass (dx,dy,dz) in 3D coordinates
        :param joints3D: 3D joint coordinates, cropped to old CoM
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, new center of mass in image coordinates
        """

        # if offset is 0, nothing to do
        if np.allclose(off, 0.):
            return dpt, joints3D, com, M

        # add offset to com
        new_com = self.joint3DToImg(self.jointImgTo3D(com, paras) + off, paras)

        # check for 1/0.
        if not (np.allclose(com[2], 0.) or np.allclose(new_com[2], 0.)):
            # scale to original size
            Mnew = self.comToTransform(new_com, cube, dpt.shape, paras)

            if len(dpt[dpt > 0]) > 0:
                new_dpt = self.recropHand(dpt, Mnew, np.linalg.inv(M), dpt.shape, paras, background_value=pad_value,
                                          nv_val=np.min(dpt[dpt > 0]) - 1, thresh_z=thresh_z, com=new_com, size=cube)
            else:
                new_dpt = self.recropHand(dpt, Mnew, np.linalg.inv(M), dpt.shape, paras, background_value=pad_value,
                                          nv_val=- 1, thresh_z=thresh_z, com=new_com, size=cube)
            # new_dpt = self.recropHand(dpt, Mnew, np.linalg.inv(M), dpt.shape, paras, background_value=pad_value,
            #                           nv_val=np.min(dpt[dpt>0])-1, thresh_z=thresh_z, com=new_com, size=cube)
        else:
            Mnew = M
            new_dpt = dpt

        # adjust joint positions to new CoM
        new_joints3D = joints3D + self.jointImgTo3D(com, paras) - self.jointImgTo3D(new_com, paras)

        return new_dpt, new_joints3D, new_com, Mnew

    def rotateHand(self, dpt, cube, com, rot, joints3D, paras=None, pad_value=0, thresh_z=True):
        """
        Rotate hand virtually in the image plane by a given angle
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param rot: rotation angle in deg
        :param joints3D: original joint coordinates, in 3D coordinates (x,y,z)
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, rotation angle in XXX
        """

        # if rot is 0, nothing to do
        if np.allclose(rot, 0.):
            return dpt, joints3D, rot

        rot = np.mod(rot, 360)

        M = cv2.getRotationMatrix2D((dpt.shape[1] // 2, dpt.shape[0] // 2), -rot, 1)
        flags = cv2.INTER_NEAREST
        new_dpt = cv2.warpAffine(dpt, M, (dpt.shape[1], dpt.shape[0]), flags=flags,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=pad_value)

        if thresh_z and len(dpt[dpt > 0]) > 0:
            new_dpt[new_dpt < (np.min(dpt[dpt > 0]) - 1)] = 0

        com3D = self.jointImgTo3D(com, paras)
        joint_2D = self.joint3DToImg(joints3D + com3D, paras)
        data_2D = np.zeros_like(joint_2D)
        for k in xrange(data_2D.shape[0]):
            data_2D[k] = rotatePoint2D(joint_2D[k], com[0:2], rot)
        new_joints3D = (self.jointImgTo3D(data_2D, paras) - com3D)

        return new_dpt, new_joints3D, rot

    def scaleHand(self, dpt, cube, com, sc, joints3D, M, paras, pad_value=0, thresh_z=True):
        """
        Virtually scale the hand by applying different cube
        :param dpt: cropped depth image with different CoM
        :param cube: metric cube of size (sx,sy,sz)
        :param com: original center of mass, in image coordinates (x,y,z)
        :param sc: scale factor for cube
        :param joints3D: 3D joint coordinates, cropped to old CoM
        :param pad_value: value of padding
        :return: adjusted image, new 3D joint coordinates, new center of mass in image coordinates
        """

        # if scale is 1, nothing to do
        if np.allclose(sc, 1.):
            return dpt, joints3D, cube, M

        new_cube = [s * sc for s in cube]

        # check for 1/0.
        if not np.allclose(com[2], 0.):
            # scale to original size
            Mnew = self.comToTransform(com, new_cube, dpt.shape, paras)
            if len(dpt[dpt > 0]) > 0:
                new_dpt = self.recropHand(dpt, Mnew, np.linalg.inv(M), dpt.shape, paras, background_value=pad_value,
                                          nv_val=np.min(dpt[dpt > 0]) - 1, thresh_z=thresh_z, com=com, size=cube)
            else:
                new_dpt = self.recropHand(dpt, Mnew, np.linalg.inv(M), dpt.shape, paras, background_value=pad_value,
                                          nv_val=- 1, thresh_z=thresh_z, com=com, size=cube)
        else:
            Mnew = M
            new_dpt = dpt

        new_joints3D = joints3D

        return new_dpt, new_joints3D, new_cube, Mnew

    def rand_augment(self, sigma_com=None, sigma_sc=None, rot_range=None):
        if sigma_com is None:
            sigma_com = 35.

        if sigma_sc is None:
            sigma_sc = 0.05

        if rot_range is None:
            rot_range = 180.

        # mode = self.rng.randint(0, len(self.aug_modes))
        # off = self.rng.randn(3) * sigma_com  # +-px/mm
        # rot = self.rng.uniform(-rot_range, rot_range)
        # sc = abs(1. + self.rng.randn() * sigma_sc)
        #
        # mode = np.random.randint(0, len(self.aug_modes))
        # off = np.random.randn(3) * sigma_com  # +-px/mm
        # rot = np.random.uniform(-rot_range, rot_range)
        # sc = abs(1. + np.random.randn() * sigma_sc)

        mode = random.randint(0, len(self.aug_modes) - 1)
        off = np.array([random.uniform(-1, 1) for a in range(3)]) * sigma_com  # +-px/mm
        rot = random.uniform(-rot_range, rot_range)
        sc = abs(1. + random.uniform(-1, 1) * sigma_sc)
        return mode, off, rot, sc

    def augmentCrop_RGB(self, img, gt3Dcrop, com, cube, M, mode, off, rot, sc, paras=None, normZeroOne=False):
        """
        Commonly used function to augment hand poses
        :param img: image
        :param gt3Dcrop: 3D annotations
        :param com: center of mass in image coordinates (x,y,z)
        :param cube: cube
        :param aug_modes: augmentation modes
        :param hd: hand detector
        :param normZeroOne: normalization
        :param sigma_com: sigma of com noise
        :param sigma_sc: sigma of scale noise
        :param rot_range: rotation range in degrees
        :return: image, 3D annotations(unnormal), com(image coordinates), cube
        """
        # assert len(img.shape) == 3
        assert isinstance(self.aug_modes, list)
        # premax = img.max()

        if self.aug_modes[mode] == 'com':
            rot = 0.
            sc = 1.
            imgRGB, new_joints3D, com, M = self.moveCoM(img.astype('float32'), cube, com, off, gt3Dcrop, M, paras,
                                                        pad_value=0, thresh_z=False)
        elif self.aug_modes[mode] == 'rot':
            off = np.zeros((3,))
            sc = 1.
            imgRGB, new_joints3D, rot = self.rotateHand(img.astype('float32'), cube, com, rot, gt3Dcrop, paras,
                                                        pad_value=0, thresh_z=False)
        elif self.aug_modes[mode] == 'sc':
            off = np.zeros((3,))
            rot = 0.
            imgRGB, new_joints3D, cube, M = self.scaleHand(img.astype('float32'), cube, com, sc, gt3Dcrop, M, paras,
                                                           pad_value=0, thresh_z=False)
        elif self.aug_modes[mode] == 'none':
            off = np.zeros((3,))
            sc = 1.
            rot = 0.
            imgRGB = img
            new_joints3D = gt3Dcrop
        else:
            raise NotImplementedError()

        # cv2.imwrite('./example_vis/0_rgb.png', imgRGB.astype(np.uint8))

        return imgRGB, None, new_joints3D, np.asarray(cube), com, M, rot

    def augmentCrop(self, img, gt3Dcrop, com, cube, M, mode, off, rot, sc, paras=None, normZeroOne=False):
        """
        Commonly used function to augment hand poses
        :param img: image
        :param gt3Dcrop: 3D annotations
        :param com: center of mass in image coordinates (x,y,z)
        :param cube: cube
        :param aug_modes: augmentation modes
        :param hd: hand detector
        :param normZeroOne: normalization
        :param sigma_com: sigma of com noise
        :param sigma_sc: sigma of scale noise
        :param rot_range: rotation range in degrees
        :return: image, 3D annotations(unnormal), com(image coordinates), cube
        """
        assert len(img.shape) == 2
        assert isinstance(self.aug_modes, list)
        premax = img.max()
        if np.max(img) == 0:
            imgD = img
            new_joints3D = gt3Dcrop
        elif self.aug_modes[mode] == 'com':
            rot = 0.
            sc = 1.
            imgD, new_joints3D, com, M = self.moveCoM(img.astype('float32'), cube, com, off, gt3Dcrop, M, paras,
                                                      pad_value=0)
        elif self.aug_modes[mode] == 'rot':
            off = np.zeros((3,))
            sc = 1.
            imgD, new_joints3D, rot = self.rotateHand(img.astype('float32'), cube, com, rot, gt3Dcrop, paras,
                                                      pad_value=0)
        elif self.aug_modes[mode] == 'sc':
            off = np.zeros((3,))
            rot = 0.
            imgD, new_joints3D, cube, M = self.scaleHand(img.astype('float32'), cube, com, sc, gt3Dcrop, M, paras,
                                                         pad_value=0)
        elif self.aug_modes[mode] == 'none':
            off = np.zeros((3,))
            sc = 1.
            rot = 0.
            imgD = img
            new_joints3D = gt3Dcrop
        else:
            raise NotImplementedError()
        imgD = self.normalize_img(premax, imgD, com, cube)
        return imgD, None, new_joints3D, np.asarray(cube), com, M, rot

    def normalize_img(self, premax, imgD, com, cube):
        imgD[imgD == premax] = com[2] + (cube[2] / 2.)
        imgD[imgD == 0] = com[2] + (cube[2] / 2.)
        imgD[imgD >= com[2] + (cube[2] / 2.)] = com[2] + (cube[2] / 2.)
        imgD[imgD <= com[2] - (cube[2] / 2.)] = com[2] - (cube[2] / 2.)
        imgD -= com[2]
        imgD /= (cube[2] / 2.)
        return imgD

    # def set_rgb_bg(self, imgRGB, imgD):
    #     # imgD[imgD == premax] = com[2] + (cube[2] / 2.)
    #     # imgD[imgD == 0] = com[2] + (cube[2] / 2.)
    #     imgRGB[imgD == 1][:] = 255
    #     # imgRGB[imgD <= com[2] - (cube[2] / 2.)] = []
    #
    #     return imgRGB

    def Crop_Image_deep_pp_RGB(self, depth, com, size, dsize, paras):
        """
        Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :param dsize: (x,y) extent of the destination size
        :return: cropped hand image, transformation matrix for joints, CoM in image coordinates
        """

        # print com, self.importer.jointImgTo3D(com)
        # import matplotlib.pyplot as plt
        # import matplotlib
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(self.dpt, cmap=matplotlib.cm.jet)

        if len(size) != 3:
            raise ValueError("Size must be 3D and bounding box")

        # calculate boundaries
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size, paras)

        # crop patch from source
        cropped = self.getCrop(depth, xstart, xend, ystart, yend, zstart, zend, thresh_z=False)

        # resize to same size
        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            sz = (dsize[0], int(hb * dsize[0] / wb))
        else:
            sz = (int(wb * dsize[1] / hb), dsize[1])

        trans = np.eye(3)
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart

        if cropped.shape[0] > cropped.shape[1]:
            scale = np.eye(3) * sz[1] / float(cropped.shape[0])
        else:
            scale = np.eye(3) * sz[0] / float(cropped.shape[1])

        scale[2, 2] = 1

        # depth resize
        rz = cv2.resize(cropped, sz, interpolation=cv2.INTER_NEAREST)

        rgb_size = (dsize[0], dsize[1], 3)
        ret = np.ones(rgb_size, np.float32) * 0  # use background as filler
        xstart = int(np.floor(dsize[0] / 2. - rz.shape[1] / 2.))
        xend = int(xstart + rz.shape[1])
        ystart = int(np.floor(dsize[1] / 2. - rz.shape[0] / 2.))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz
        # print rz.shape, xstart, ystart
        off = np.eye(3)
        off[0, 2] = xstart
        off[1, 2] = ystart

        return ret, np.dot(off, np.dot(scale, trans))

    # use deep-pp's method
    def Crop_Image_deep_pp(self, depth, com, size, dsize, paras):
        """
        Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :param dsize: (x,y) extent of the destination size
        :return: cropped hand image, transformation matrix for joints, CoM in image coordinates
        """

        # print com, self.importer.jointImgTo3D(com)
        # import matplotlib.pyplot as plt
        # import matplotlib
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(self.dpt, cmap=matplotlib.cm.jet)

        if len(size) != 3 or len(dsize) != 2:
            raise ValueError("Size must be 3D and dsize 2D bounding box")

        # calculate boundaries
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size, paras)

        # crop patch from source
        cropped = self.getCrop(depth, xstart, xend, ystart, yend, zstart, zend)

        # resize to same size
        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            sz = (dsize[0], int(hb * dsize[0] / wb))
        else:
            sz = (int(wb * dsize[1] / hb), dsize[1])

        trans = np.eye(3)
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart

        if cropped.shape[0] > cropped.shape[1]:
            scale = np.eye(3) * sz[1] / float(cropped.shape[0])
        else:
            scale = np.eye(3) * sz[0] / float(cropped.shape[1])

        scale[2, 2] = 1

        # depth resize
        rz = cv2.resize(cropped, sz, interpolation=cv2.INTER_NEAREST)

        ret = np.ones(dsize, np.float32) * 0  # use background as filler
        xstart = int(np.floor(dsize[0] / 2. - rz.shape[1] / 2.))
        xend = int(xstart + rz.shape[1])
        ystart = int(np.floor(dsize[1] / 2. - rz.shape[0] / 2.))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz
        # print rz.shape, xstart, ystart
        off = np.eye(3)
        off[0, 2] = xstart
        off[1, 2] = ystart

        return ret, np.dot(off, np.dot(scale, trans))

    def getCrop(self, depth, xstart, xend, ystart, yend, zstart, zend, thresh_z=True, background=0):
        """
        Crop patch from image
        :param depth: depth image to crop from
        :param xstart: start x
        :param xend: end x
        :param ystart: start y
        :param yend: end y
        :param zstart: start z
        :param zend: end z
        :param thresh_z: threshold z values
        :return: cropped image
        """
        if len(depth.shape) == 2:
            cropped = depth[max(ystart, 0):min(yend, depth.shape[0]), max(xstart, 0):min(xend, depth.shape[1])].copy()
            # add pixels that are out of the image in order to keep aspect ratio
            cropped = np.pad(cropped, ((abs(ystart) - max(ystart, 0),
                                        abs(yend) - min(yend, depth.shape[0])),
                                       (abs(xstart) - max(xstart, 0),
                                        abs(xend) - min(xend, depth.shape[1]))), mode='constant',
                             constant_values=background)
        elif len(depth.shape) == 3:
            cropped = depth[max(ystart, 0):min(yend, depth.shape[0]), max(xstart, 0):min(xend, depth.shape[1]),
                      :].copy()
            # add pixels that are out of the image in order to keep aspect ratio
            cropped = np.pad(cropped, ((abs(ystart) - max(ystart, 0),
                                        abs(yend) - min(yend, depth.shape[0])),
                                       (abs(xstart) - max(xstart, 0),
                                        abs(xend) - min(xend, depth.shape[1])),
                                       (0, 0)), mode='constant', constant_values=background)
        else:
            raise NotImplementedError()

        if thresh_z is True:
            msk1 = np.logical_and(cropped < zstart, cropped != 0)
            msk2 = np.logical_and(cropped > zend, cropped != 0)
            cropped[msk1] = zstart
            cropped[msk2] = 0.  # backface is at 0, it is set later
        return cropped

    # tensor
    def unnormal_joint_img(self, joint_img):
        device = joint_img.device
        joint = torch.zeros(joint_img.size()).to(device)
        joint[:, :, 0:2] = (joint_img[:, :, 0:2] + 1) / 2 * self.img_size
        joint[:, :, 2] = (joint_img[:, :, 2] + 1) / 2 * self.cube_size[2]
        return joint

    def uvd_nl2xyz_tensor(self, uvd, center, m, cube, cam_paras):
        batch_size, point_num, _ = uvd.size()
        device = uvd.device
        cube_size_t = cube.to(device).view(batch_size, 1, 3).repeat(1, point_num, 1)
        center_t = center.to(device).view(batch_size, 1, 3).repeat(1, point_num, 1)
        M_t = m.to(device).view(batch_size, 1, 3, 3)
        M_inverse = torch.linalg.inv(M_t).repeat(1, point_num, 1, 1)

        uv_unnormal = (uvd[:, :, 0:2] + 1) * (self.img_size / 2)
        d_unnormal = (uvd[:, :, 2:]) * (cube_size_t[:, :, 2:] / 2.0) + center_t[:, :, 2:]
        uvd_unnormal = torch.cat((uv_unnormal, d_unnormal), dim=-1)
        uvd_world = self.get_trans_points(uvd_unnormal, M_inverse)
        xyz = self.pointsImgTo3D(uvd_world, cam_paras)
        return xyz

    def uvd_nl2xyznl_tensor(self, uvd, center, m, cube, cam_paras):
        batch_size, point_num, _ = uvd.size()
        device = uvd.device
        cube_size_t = cube.to(device).view(batch_size, 1, 3).repeat(1, point_num, 1)
        center_t = center.to(device).view(batch_size, 1, 3).repeat(1, point_num, 1)
        M_t = m.to(device).view(batch_size, 1, 3, 3)
        M_inverse = torch.linalg.inv(M_t).repeat(1, point_num, 1, 1)

        uv_unnormal = (uvd[:, :, 0:2] + 1) * (self.img_size / 2)
        d_unnormal = (uvd[:, :, 2:]) * (cube_size_t[:, :, 2:] / 2.0) + center_t[:, :, 2:]
        uvd_unnormal = torch.cat((uv_unnormal, d_unnormal), dim=-1)
        uvd_world = self.get_trans_points(uvd_unnormal, M_inverse)
        xyz = self.pointsImgTo3D(uvd_world, cam_paras)
        xyz_noraml = (xyz - center_t) / (cube_size_t / 2.0)
        return xyz_noraml

    def img2anchor_dis(self, joint_uvd, img, center, M, cube, cam_para, gamma=10):
        '''
        :param img: Bx1xWxH Tensor
        :param feature: BxCxWxH Tensor
        :return: select_feature: BxCxN
        '''
        device = img.device
        B, J, _ = joint_uvd.size()
        B, _, W, H = img.size()
        joint_xyz = self.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para)
        mesh_x = 2.0 * (torch.arange(W).unsqueeze(1).expand(W, W).float() + 0.5) / W - 1.0
        mesh_y = 2.0 * (torch.arange(W).unsqueeze(0).expand(W, W).float() + 0.5) / W - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(B, 1, 1, 1).to(device)
        img_uvd = torch.cat((coords, img), dim=1).view(B, 3, H * W).permute(0, 2, 1)

        # img_uvd[img_uvd[:, :,2] == 1] = 10

        # B*N*3
        img_xyz = self.uvd_nl2xyznl_tensor(img_uvd, center, M, cube, cam_para).unsqueeze(1).repeat(1, J, 1, 1)

        # B*j*N*3
        # distance = torch.sqrt(torch.sum(torch.pow(pcl.unsqueeze(2) - img_xyz.unsqueeze(1), 2), dim=-1) + 1e-8)
        distance = torch.sum(torch.pow(img_xyz - joint_xyz.unsqueeze(2), 2), dim=-1)

        # closeness_value = 1 / (distance + 1e-8)
        # closeness_value_normal = closeness_value / (closeness_value.sum(-1, keepdim=True) + 1e-8)
        closeness_value_normal = 1 / (gamma * distance + 1)
        return closeness_value_normal.view(B, J, H, W)

    def xyz_nl2uvdnl_tensor(self, joint_xyz, center, M, cube_size, cam_paras):
        device = joint_xyz.device
        batch_size, joint_num, _ = joint_xyz.size()
        cube_size_t = cube_size.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
        center_t = center.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
        M_t = M.to(device).view(batch_size, 1, 3, 3).repeat(1, joint_num, 1, 1)

        joint_temp = joint_xyz * cube_size_t / 2.0 + center_t
        joint_uvd = self.points3DToImg(joint_temp, cam_paras)
        joint_uvd = self.get_trans_points(joint_uvd, M_t)
        joint_uv = joint_uvd[:, :, 0:2] / self.img_size * 2.0 - 1
        joint_d = (joint_uvd[:, :, 2:] - center_t[:, :, 2:]) / (cube_size_t[:, :, 2:] / 2)
        joint = torch.cat((joint_uv, joint_d), dim=-1)
        return joint

    def get_trans_points(self, joints, M):
        device = joints.device
        joints_mat = torch.cat((joints[:, :, 0:2], torch.ones(joints.size(0), joints.size(1), 1).to(device)),
                               dim=-1).float()
        joints_trans_xy = torch.matmul(M, joints_mat.unsqueeze(-1)).squeeze(-1)[:, :, 0:2]
        joints_trans_z = joints[:, :, 2:]
        return torch.cat((joints_trans_xy, joints_trans_z), dim=-1)

    def getpcl(self, imgD, com3D, cube, M, cam_para=None):
        mask = np.isclose(imgD, 1)
        dpt_ori = imgD * cube[2] / 2.0 + com3D[2]
        # change the background value
        dpt_ori[mask] = 0

        pcl = (self.depthToPCL(dpt_ori, M, cam_para) - com3D)
        pcl_num = pcl.shape[0]
        cube_tile = np.tile(cube / 2.0, pcl_num).reshape([pcl_num, 3])
        pcl = pcl / cube_tile
        return pcl

    def farthest_point_sample(self, xyz, npoint):
        N, C = xyz.shape
        S = npoint
        if N < S:
            centroids = np.arange(N)
            centroids = np.append(centroids, np.random.choice(centroids, size=S - N, replace=False))
        else:
            centroids = np.zeros(S).astype(np.int)
            distance = np.ones(N) * 1e10
            farthest = np.random.randint(0, S)
            for i in range(S):
                centroids[i] = farthest
                centroid = xyz[farthest, :]
                dist = np.sum((xyz - centroid) ** 2, -1)
                mask = dist < distance
                distance[mask] = dist[mask]
                farthest = distance.argmax()
        return np.unique(centroids)

    def depthToPCL(self, dpt, T, paras=None, background_val=0.):
        if isinstance(paras, tuple):
            fx, fy, fu, fv = paras
        else:
            fx, fy, fu, fv = self.paras
        # get valid points and transform
        pts = np.asarray(np.where(~np.isclose(dpt, background_val))).transpose()
        pts = np.concatenate([pts[:, [1, 0]] + 0.5, np.ones((pts.shape[0], 1), dtype='float32')], axis=1)
        pts = np.dot(np.linalg.inv(np.asarray(T)), pts.T).T
        pts = (pts[:, 0:2] / pts[:, 2][:, None]).reshape((pts.shape[0], 2))

        # replace the invalid data
        depth = dpt[(~np.isclose(dpt, background_val))]

        # get x and y data in a vectorized way
        row = (pts[:, 0] - fu) / fx * depth
        col = self.flip * (pts[:, 1] - fv) / fy * depth

        # combine x,y,depth
        return np.column_stack((row, col, depth))

    def pca_point(self, pcl, joint):
        self.pca.fit(pcl)
        coeff = self.pca.components_.T
        # if coeff[1, 0] < 0:
        #     coeff[:, 0] = -coeff[:, 0]
        # if coeff[2, 2] < 0:
        #     coeff[:, 2] = -coeff[:, 2]
        coeff[:, 1] = np.cross(coeff[:, 2], coeff[:, 0])
        points_rotation = np.dot(pcl, coeff)
        joint_rotation = np.dot(joint, coeff)
        return points_rotation, joint_rotation, coeff

    def img2pcl_index_softmax(self, pcl, img, center, M, cube, cam_para, select_num=64, scale=30):
        '''
        :param pcl: BxNx3 Tensor
        :param img: Bx1xWxH Tensor
        :param feature: BxCxWxH Tensor
        :return: select_feature: BxCxN
        '''

        device = pcl.device
        B, N, _ = pcl.size()
        B, _, W, H = img.size()

        mesh_x = 2.0 * (torch.arange(W).unsqueeze(1).expand(W, W).float() + 0.5) / W - 1.0
        mesh_y = 2.0 * (torch.arange(W).unsqueeze(0).expand(W, W).float() + 0.5) / W - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(B, 1, 1, 1).to(device)
        img_uvd = torch.cat((coords, img), dim=1).view(B, 3, H * W).permute(0, 2, 1)
        img_xyz = self.uvd_nl2xyznl_tensor(img_uvd, center, M, cube, cam_para)

        distance = torch.sum(torch.pow(pcl.unsqueeze(2) - img_xyz.unsqueeze(1), 2), dim=-1)
        distance_value, distance_index = torch.topk(distance, select_num, largest=False)

        distance_value = torch.sqrt(distance_value + 1e-8)
        distance_value = distance_value - distance_value.min(dim=-1, keepdim=True)[0]
        closeness_value = 1 - distance_value / distance_value.max(dim=-1, keepdim=True)[0]
        # closeness_value = torch.clamp(0.1 - distance_value, 0, 1)
        closeness_value_normal = torch.softmax(closeness_value * scale, dim=-1)
        return closeness_value_normal, distance_index

    def img2pcl_index(self, pcl, img, center, M, cube, cam_para, select_num=9):
        '''
        :param pcl: BxNx3 Tensor
        :param img: Bx1xWxH Tensor
        :param feature: BxCxWxH Tensor
        :return: select_feature: BxCxN
        '''

        device = pcl.device
        B, N, _ = pcl.size()
        B, _, W, H = img.size()

        mesh_x = 2.0 * (torch.arange(W).unsqueeze(1).expand(W, W).float() + 0.5) / W - 1.0
        mesh_y = 2.0 * (torch.arange(W).unsqueeze(0).expand(W, W).float() + 0.5) / W - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(B, 1, 1, 1).to(device)
        img_uvd = torch.cat((coords, img), dim=1).view(B, 3, H * W).permute(0, 2, 1)
        img_xyz = self.uvd_nl2xyznl_tensor(img_uvd, center, M, cube, cam_para)

        # distance = torch.sqrt(torch.sum(torch.pow(pcl.unsqueeze(2) - img_xyz.unsqueeze(1), 2), dim=-1) + 1e-8)
        distance = torch.sum(torch.pow(pcl.unsqueeze(2) - img_xyz.unsqueeze(1), 2), dim=-1)
        distance_value, distance_index = torch.topk(distance, select_num, largest=False)
        # version 1
        closeness_value = 1 / (distance_value + 1e-8)
        closeness_value_normal = closeness_value / (closeness_value.sum(-1, keepdim=True) + 1e-8)

        # version 2
        # distance_value = torch.sqrt(distance_value + 1e-8)
        # distance_value = distance_value - distance_value.min(dim=-1,keepdim=True)[0]
        # closeness_value = 1 - distance_value / distance_value.max(dim=-1,keepdim=True)[0]
        # closeness_value_normal = torch.softmax(closeness_value*30, dim=-1)
        return closeness_value_normal, distance_index

    def pcl2img_index(self, pcl, img_size, center, M, cube, cam_para, select_num=9):
        '''
        :param pcl: BxNx3 Tensor
        :param img: Bx1xWxH Tensor
        :param feature: BxCxWxH Tensor
        :return: select_feature: BxCxN
        '''

        device = pcl.device
        B, N, _ = pcl.size()

        pcl_uvd = (self.xyz_nl2uvdnl_tensor(pcl, center, M, cube, cam_para)[:, :, :2] + 1) / 2 * img_size
        mesh_x = (torch.arange(img_size).unsqueeze(1).expand(img_size, img_size).float() + 0.5)
        mesh_y = (torch.arange(img_size).unsqueeze(0).expand(img_size, img_size).float() + 0.5)
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(B, 1, 1, 1).to(device)
        coords = coords.view(B, 2, img_size * img_size).permute(0, 2, 1)

        distance = torch.sqrt(torch.sum(torch.pow(pcl_uvd.unsqueeze(2) - coords.unsqueeze(1), 2), dim=-1) + 1e-8)
        distance_value, distance_index = torch.topk(distance, select_num, largest=False)
        closeness_value = 1 / (distance_value + 1e-8)
        closeness_value_normal = closeness_value / (closeness_value.sum(-1, keepdim=True) + 1e-8)
        return closeness_value_normal, distance_index

    def img2pcl(self, img):
        '''
        :param img: Bx1xWxH Tensor
        '''
        device = img.device
        B, _, W, H = img.size()

        mesh_x = 2.0 * (torch.arange(W).unsqueeze(1).expand(W, W).float() + 0.5) / W - 1.0
        mesh_y = 2.0 * (torch.arange(W).unsqueeze(0).expand(W, W).float() + 0.5) / W - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(B, 1, 1, 1).to(device)
        img_uvd = torch.cat((coords, img), dim=1).view(B, 3, H * W).permute(0, 2, 1)
        return img_uvd

    """
    pcl: B N 3
    joint: B J 3 
    """

    # def occlusion(self, pcl, joint):
    #     B, J, _ = joint.size()
    #     offset = joint.unsqueeze(2) - pcl.unsqueeze(1)
    #     dis = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=-1))
    #     dis = torch.mean(torch.topk(dis, 10, largest=False)[0], dim=-1)
    #     joint_kernel = self.joint_kernel.to(pcl.device).unsqueeze(0).repeat(B, 1)
    #     return dis.lt(joint_kernel)

    def visible(self, pcl, joint):
        B, J, _ = joint.size()
        offset = joint.unsqueeze(2) - pcl.unsqueeze(1)
        dis = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=-1) + 1e-8)
        joint_kernel = self.joint_kernel.to(pcl.device).view(1, -1, 1)
        heatmap = (joint_kernel - dis) / joint_kernel
        visible = heatmap.gt(0).sum(-1).gt(10)
        return visible

    def offset2visible(self, offset):
        B, _, H, W = offset.size()
        heatmap = offset[:, self.joint_num * 3:self.joint_num * 4, :, :]
        return heatmap.gt(0).sum(-1).sum(-1).gt(10)

    def mask_img(self, img, img_joint_uvd, mask_offset, mask_para, min_mask_num=3, max_mask_num=10):
        device = img.device
        S = img.size(-1)
        mask_num = np.random.choice(np.arange(min_mask_num, max_mask_num), 1, replace=False)[0]
        b, j, _ = img_joint_uvd.size()
        joint_id = np.random.choice(np.arange(0, j), mask_num, replace=False)
        mask_uvd = img_joint_uvd[:, joint_id, :]
        uvd_offset = (torch.rand(mask_uvd.size()) - 0.5) * mask_offset * 2
        mask_uvd = mask_uvd + uvd_offset.to(device)
        mask_range = torch.rand([b, mask_num]).to(device) * mask_para
        xx, yy = np.meshgrid(np.arange(S), np.arange(S))
        xx = 2 * (xx + 0.5) / S - 1.0
        yy = 2 * (yy + 0.5) / S - 1.0
        mesh = np.stack((xx, yy), axis=-1).reshape([1, -1, 2])
        mesh = torch.from_numpy(mesh).float().to(device).view(1, -1, 2).repeat(b, 1, 1)
        mesh = torch.cat((mesh, img.view(b, -1, 1)), dim=-1).view(b, 1, -1, 3)
        dis = torch.sqrt(torch.sum((mesh - mask_uvd.view(b, mask_num, 1, 3)) ** 2, dim=-1))
        mask = dis.lt(mask_range.view([b, mask_num, 1])).float()
        mask = ~mask.sum(1).gt(0)
        return torch.where(mask.view(b, 1, img.size(-2), img.size(-1)), img, torch.ones_like(img).to(device))


def project_3D_points(cam_mat, pts3D, is_OpenGL_coords=True):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if is_OpenGL_coords:
        pts3D = pts3D.dot(coord_change_mat.T)

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:, 0] / proj_pts[:, 2], proj_pts[:, 1] / proj_pts[:, 2]], axis=1)

    assert len(proj_pts.shape) == 2

    return proj_pts


class HO3D(loader):
    def __init__(self):
        super(HO3D, self).__init__()

        self.transform = transforms.ToTensor()
        # self.input_modal = input_modal

    def augmentation_depth(self, depth, cube, center_uvd, cam_para):

        # center_uvd = self.joint3DToImg(center_xyz, cam_para)
        depth_crop, trans = self.Crop_Image_deep_pp(copy.deepcopy(depth), center_uvd, self.cube_size,
                                                    (self.img_size, self.img_size), cam_para)

        imgD = self.normalize_img(depth_crop.max(), depth_crop, center_uvd, self.cube_size)

        com2D = center_uvd
        M = trans
        com3D = self.jointImgTo3D(com2D, cam_para)

        # get pcl
        pcl = self.getpcl(imgD, com3D, cube, M, cam_para)
        pcl_index = np.arange(pcl.shape[0])
        pcl_num = pcl.shape[0]
        if pcl_num == 0:
            pcl_sample = np.zeros([self.sample_num, 3])
        else:
            if pcl_num < self.sample_num:
                tmp = math.floor(self.sample_num / pcl_num)
                index_temp = pcl_index.repeat(tmp)
                pcl_index = np.append(index_temp, np.random.choice(pcl_index, size=divmod(self.sample_num, pcl_num)[1],
                                                                   replace=False))
            select = np.random.choice(pcl_index, self.sample_num, replace=False)
            pcl_sample = pcl[select, :]
        pcl_sample = torch.from_numpy(pcl_sample).float()
        pcl_sample = torch.clamp(pcl_sample, -1, 1)

        data_depth = torch.from_numpy(imgD).float()
        data_depth = data_depth.unsqueeze(0)

        M = torch.from_numpy(M).float()

        return data_depth, pcl_sample, M

    def generatePCL(self, imgD, com3D, cube, M, cam_para):
        # get pcl

        pcl = self.getpcl(imgD, com3D, cube, M, cam_para)
        pcl_index = np.arange(pcl.shape[0])
        pcl_num = pcl.shape[0]
        if pcl_num == 0:
            pcl_sample = np.zeros([self.sample_num, 3])
        else:
            if pcl_num < self.sample_num:
                tmp = math.floor(self.sample_num / pcl_num)
                index_temp = pcl_index.repeat(tmp)
                pcl_index = np.append(index_temp, np.random.choice(pcl_index, size=divmod(self.sample_num, pcl_num)[1],
                                                                   replace=False))
            select = np.random.choice(pcl_index, self.sample_num, replace=False)
            pcl_sample = pcl[select, :]
        pcl_sample = torch.from_numpy(pcl_sample).float()
        pcl_sample = torch.clamp(pcl_sample, -1, 1)

    def process_bbox(self, bbox, img_width, img_height, expansion_factor=1.25):
        # sanitize bboxes
        x, y, w, h = bbox
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
        if w * h > 0 and x2 >= x1 and y2 >= y1:
            bbox = np.array([x1, y1, x2 - x1, y2 - y1])
        else:
            return None

        # aspect ratio preserving bbox
        w = bbox[2]
        h = bbox[3]
        c_x = bbox[0] + w / 2.
        c_y = bbox[1] + h / 2.
        aspect_ratio = 1
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        bbox[2] = w * expansion_factor
        bbox[3] = h * expansion_factor
        bbox[0] = c_x - bbox[2] / 2.
        bbox[1] = c_y - bbox[3] / 2.

        return bbox


def draw_pcl(verts, batch_index, data_dir, img_name):
    batch_size = verts.size(0)
    verts = verts.detach().cpu().numpy()
    for index in range(batch_size):
        path = data_dir + '/' + str(batch_index * batch_size + index) + '_' + img_name + '.obj'
        with open(path, 'w') as fp:
            for v in verts[index]:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))


def convert_nyu2mano(joint):
    select_joint = joint.clone()
    select_joint[:, 1, :] = joint[:, 1, :] + (joint[:, 2, :] - joint[:, 1, :]) * 0.3
    select_joint[:, 5, :] = joint[:, 5, :] + (joint[:, 6, :] - joint[:, 5, :]) * 0.3
    select_joint[:, 9, :] = joint[:, 9, :] + (joint[:, 10, :] - joint[:, 9, :]) * 0.3
    select_joint[:, 13, :] = joint[:, 13, :] + (joint[:, 14, :] - joint[:, 13, :]) * 0.3
    select_joint[:, 17, :] = joint[:, 17, :] + (joint[:, 18, :] - joint[:, 17, :]) * 0.2

    select_joint[:, 0, :] = joint[:, 0, :] - (joint[:, 1, :] - joint[:, 0, :]) * 0.3
    select_joint[:, 4, :] = joint[:, 4, :] - (joint[:, 5, :] - joint[:, 4, :]) * 0.3
    select_joint[:, 8, :] = joint[:, 8, :] - (joint[:, 9, :] - joint[:, 8, :]) * 0.3
    select_joint[:, 12, :] = joint[:, 12, :] - (joint[:, 13, :] - joint[:, 12, :]) * 0.3
    select_joint[:, 16, :] = joint[:, 16, :] - (joint[:, 17, :] - joint[:, 16, :]) * 0.3

    select_joint[:, 3, :] = joint[:, 3, :] - (joint[:, 3, :] - joint[:, 2, :]) * 0.1
    select_joint[:, 7, :] = joint[:, 7, :] - (joint[:, 7, :] - joint[:, 6, :]) * 0.1
    select_joint[:, 11, :] = joint[:, 11, :] - (joint[:, 11, :] - joint[:, 10, :]) * 0.2
    select_joint[:, 15, :] = joint[:, 15, :] - (joint[:, 15, :] - joint[:, 14, :]) * 0.3

    NYU2MANO = [22,
                15, 14, 13,
                11, 10, 9,
                3, 2, 1,
                7, 6, 5,
                19, 18, 17,
                12, 8, 0, 4, 16]
    return select_joint[:, NYU2MANO, :]


def display_sphere(centers, radiuss, ax=None, transpose=True):
    num = centers.shape[0]
    if transpose:
        centers = centers[:, [0, 2, 1]]
    for index in range(num):
        center = centers[index]
        radius = radiuss[index]
        t = np.linspace(0, np.pi * 2, 20)
        s = np.linspace(0, np.pi, 20)

        t, s = np.meshgrid(t, s)
        x = np.cos(t) * np.sin(s)
        y = np.sin(t) * np.sin(s)
        z = np.cos(s)
        ax.plot_surface(x * radius + center[0], y * radius + center[1], z * radius + center[2], rstride=1, cstride=1)


if __name__ == "__main__":
    # vis_dexycb_dataset()
    # vis_ho_dataset()
    # vis_nyu_dataset()
    # debug_mano_point_sample()
    # opt_nyu_mesh()
    # vis_dexycb_hardcase()
    # vis_nyu_hardcase()
    # vis_ho3d_hardcase()
    # vis_nyu_txt()
    # hard_case_error()
    # vis_ho3d_hardcase_mesh()
    # vis_nyu_hardcase_mesh()
    # vis_dexycb_hardcase_mesh()
    trainData = DexYCBDataset('s0', 'train', '/home/cyc/pycharm/data/hand/', aug_para=[10, 0.2, 180],
                              input_modal='RGBD')
