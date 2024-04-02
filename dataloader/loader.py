# -*- coding: utf-8 -*-
import os

import cv2
import torch
import numpy as np
from PIL import Image
import os.path as osp
import copy
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import sys
import math
import scipy.io as sio
import random

sys.path.append('..')
from util import vis_tool
from scipy import ndimage
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
xrange = range


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
    def __init__(self, root_dir, phase, img_size, center_type, dataset_name):
        self.rng = np.random.RandomState(23455)
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.phase = phase
        self.img_size = img_size
        self.center_type = center_type
        self.allJoints = False
        self.pca = PCA(n_components=3)
        self.sample_num = 1024

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

    def joint3DToImg(self, xyz, paras=None, flip=1):
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

    def Crop_Image_deep_pp_RGB(self, depth, com, size, dsize, paras):
        """
        Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :param dsize: (x,y) extent of the destination size
        :return: cropped hand image, transformation matrix for joints, CoM in image coordinates
        """

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
        joints_mat = torch.cat((joints[:, :, 0:2], torch.ones(joints.size(0), joints.size(1), 1).to(device)), dim=-1)
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

    def read_modelPara(self, data_rt, view):
        theta = np.loadtxt(data_rt + '/posePara_lm_collosion/' + self.dataset_name + '-' + self.phase + '-' + str(
            view) + '-pose.txt').reshape(-1, 45)
        quat = np.loadtxt(data_rt + '/posePara_lm_collosion/' + self.dataset_name + '-' + self.phase + '-' + str(
            view) + '-glb.txt').reshape(-1, 3)
        scale = np.loadtxt(data_rt + '/posePara_lm_collosion/' + self.dataset_name + '-' + self.phase + '-' + str(
            view) + '-scale.txt').reshape(-1, 1)
        trans = np.loadtxt(data_rt + '/posePara_lm_collosion/' + self.dataset_name + '-' + self.phase + '-' + str(
            view) + '-trans.txt').reshape(-1, 3)
        shape = np.loadtxt(data_rt + '/posePara_lm_collosion/' + self.dataset_name + '-' + self.phase + '-' + str(
            view) + '-shape.txt').reshape(-1, 10)

        model_para = np.concatenate([quat, theta, shape, scale, trans], axis=-1)
        return model_para


class DexYCBDataset(loader):
    def __init__(self, setup, split, root_dir, img_size=128, aug_para=[10, 0.2, 180], input_modal='RGBD'):
        super(DexYCBDataset, self).__init__(root_dir, split, img_size, 'joint_mean', 'DexYCB')

        # self.writer = SummaryWriter('runs/DexYCB_fliter')
        self.setup = setup
        self.split = split
        self.aug_para = aug_para
        self.cube_size = [250, 250, 250]
        self.aug_modes = ['rot', 'com', 'sc', 'none']
        self.flip = 1

        self.root_dir = root_dir + '/DexYCB'
        self.annot_path = osp.join(self.root_dir, 'annotations')
        self.input_modal = input_modal
        self.transform = transforms.ToTensor()

        self.datalist = self.load_data()
        print('loading finish')
        print('len: %d' % (len(self.datalist)))

    def load_data(self):
        db = COCO(osp.join(self.annot_path, "DEX_YCB_{}_{}_data.json".format(self.setup, self.split)))
        # user_name = self.root_dir.split('/')[2]
        datalist = []

        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]

            img_path = osp.join(self.root_dir, img['color_file_name'])
            img_path = img_path.replace('/home/pfren/dataset/', '/home/cyc/pycharm/data/')
            # img_path = osp.join(self.root_dir, img['file_name'])
            img_shape = (img['height'], img['width'])

            if self.split == 'train':
                # joints_coord_cam = np.array(ann['joint_3d'], dtype=np.float32)  # meter
                joints_coord_cam = np.array(ann['joints_coord_cam'], dtype=np.float32) / 1000  # meter
                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann['cam_param'].items()}
                hand_type = ann['hand_type']

                if joints_coord_cam.sum() == -63:
                    continue

                mano_pose = np.array(ann['mano_param']['pose'], dtype=np.float32)
                mano_shape = np.array(ann['mano_param']['shape'], dtype=np.float32)
                mano_trans = np.array(ann['mano_param']['trans'], dtype=np.float32)

                data = {"img_path": img_path, "img_shape": img_shape, "joints_coord_cam": joints_coord_cam,
                        "cam_param": cam_param, "mano_pose": mano_pose, "mano_shape": mano_shape,
                        'mano_trans': mano_trans, "hand_type": hand_type}
            else:
                # joints_coord_cam = np.array(ann['joint_3d'], dtype=np.float32)
                joints_coord_cam = np.array(ann['joints_coord_cam'], dtype=np.float32) / 1000  # m
                hand_type = ann['hand_type']

                if joints_coord_cam.sum() == -63:
                    continue

                mano_pose = np.array(ann['mano_param']['pose'], dtype=np.float32)
                mano_shape = np.array(ann['mano_param']['shape'], dtype=np.float32)
                mano_trans = np.array(ann['mano_param']['trans'], dtype=np.float32)
                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann['cam_param'].items()}
                data = {"img_path": img_path, "img_shape": img_shape, "joints_coord_cam": joints_coord_cam,
                        "mano_pose": mano_pose, "mano_shape": mano_shape, 'mano_trans': mano_trans,
                        "cam_param": cam_param, "image_id": image_id, 'hand_type': hand_type}

            datalist.append(data)
        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):

        # idx = idx + 11088
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape = data['img_path'], data['img_shape']
        hand_type = data['hand_type']
        do_flip = (hand_type == 'left')
        if 'RGB' in self.input_modal:
            # rgb = load_img(img_path.replace('png', 'jpg'))
            rgb = cv2.imread(img_path)
            if not isinstance(rgb, np.ndarray):
                raise IOError("Fail to read %s" % img_path)

        depth = cv2.imread(img_path.replace('color_', 'aligned_depth_to_color_').replace('jpg', 'png'),
                           cv2.IMREAD_ANYDEPTH)

        intrinsics = data['cam_param']
        cam_para = (intrinsics['focal'][0], intrinsics['focal'][1], intrinsics['princpt'][0], intrinsics['princpt'][1])
        joint_xyz = data['joints_coord_cam'].reshape([21, 3])[DexYCB2MANO, :] * 1000
        joint_uvd = self.joint3DToImg(joint_xyz, cam_para)
        mano_pose, mano_shape, mano_trans = data['mano_pose'], data['mano_shape'], data['mano_trans']

        name_list = img_path.split('/')
        dir_1, dir_2, dir_3, dir_4 = name_list[-4], name_list[-3], name_list[-2], name_list[-1]

        if do_flip:
            if 'RGB' in self.input_modal:
                rgb = rgb[:, ::-1].copy()
            depth = depth[:, ::-1].copy()

            joint_uvd[:, 0] = img_shape[1] - joint_uvd[:, 0] - 1

        joint_xyz = self.jointImgTo3D(joint_uvd, cam_para)
        center_xyz = joint_xyz.mean(0)
        gt3Dcrop = joint_xyz - center_xyz
        center_uvd = self.joint3DToImg(center_xyz, cam_para)

        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, self.cube_size, (self.img_size, self.img_size),
                                                    cam_para)
        if 'RGB' in self.input_modal:
            rgb_crop, trans_rgb = self.Crop_Image_deep_pp_RGB(copy.deepcopy(rgb), center_uvd, self.cube_size,
                                                              (self.img_size, self.img_size), cam_para)

        if self.split == 'train':
            mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1],
                                                   rot_range=self.aug_para[2])  # 10, 0.1, 180
            imgD, _, curLabel, cube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, self.cube_size,
                                                                    trans,
                                                                    mode, off, rot, sc, cam_para)

            if 'RGB' in self.input_modal:
                imgRGB, _, curLabel_rgb, cube_rgb, com2D_rgb, M_rgb, _ = self.augmentCrop_RGB(rgb_crop, gt3Dcrop,
                                                                                              center_uvd,
                                                                                              self.cube_size, trans_rgb,
                                                                                              mode, off, rot, sc,
                                                                                              cam_para)
                # RGB normalization
                imgRGB = self.transform(imgRGB.astype(np.float32)) / 255.

            curLabel = curLabel / (cube[2] / 2.0)

        else:
            imgD = self.normalize_img(depth_crop.max(), depth_crop, center_xyz, self.cube_size)
            if 'RGB' in self.input_modal:
                imgRGB = self.transform(rgb_crop.astype(np.float32)) / 255.
            curLabel = gt3Dcrop / (self.cube_size[2] / 2.0)
            cube = np.array(self.cube_size)
            com2D = center_uvd
            M = trans

        com3D = self.jointImgTo3D(com2D, cam_para)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (cube[0] / 2.0) + com3D, cam_para), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size / 2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (cube[0] / 2.0)

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

        data_depth = torch.from_numpy(imgD).float()
        data_depth = data_depth.unsqueeze(0)

        if 'RGB' in self.input_modal:
            data_rgb = imgRGB
        else:
            data_rgb = None

        joint_img = torch.from_numpy(joint_img).float()
        joint = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(cube).float()

        cam_para = torch.from_numpy(np.array(cam_para)).float()

        return data_rgb, data_depth, pcl_sample, joint, joint_img, center, M, cube, cam_para


class HO3D(loader):
    def __init__(self, data_split, root_dir, dataset_version='v3', img_size=128, center_type='refine',
                 aug_para=[10, 0.2, 180], cube_size=[280, 280, 280], input_modal='RGBD', color_factor=0.2):
        super(HO3D, self).__init__(root_dir, data_split, img_size, center_type, 'HO3D')

        self.data_split = data_split
        self.dataset_version = dataset_version
        self.root_dir = osp.join(root_dir, 'HO3D_%s' % (dataset_version))
        self.annot_path = osp.join(self.root_dir, 'annotations')
        self.root_joint_idx = 0
        self.color_factor = color_factor

        self.aug_para = aug_para
        self.cube_size = cube_size
        self.aug_modes = ['rot', 'com', 'sc', 'none']  # 'rot', 'com', 'sc', 'none'
        self.flip = 1
        if center_type == 'refine':
            self.center_xyz = np.loadtxt(self.root_dir + '/annotations/%s_refine_center_xyz.txt' % (data_split))
        self.dataset_len = 0
        self.datalist = self.load_data()
        print('Dataset len:' + str(self.dataset_len))

        self.transform = transforms.ToTensor()
        self.input_modal = input_modal

    def load_data(self):
        db = COCO(osp.join(self.annot_path, "HO3D_{}_data.json".format(self.data_split)))
        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]

            img_shape = (img['height'], img['width'])
            if self.data_split == 'train' or self.data_split == 'test' or self.data_split == 'train_all':
                img_path = osp.join(self.root_dir, img['file_name'])
                joints_coord_cam = np.array(ann['joints_coord_cam'], dtype=np.float32).reshape([21, 3])  # meter
                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann['cam_param'].items()}
                fx, fy, fu, fv = cam_param['focal'][0], cam_param['focal'][1], cam_param['princpt'][0], \
                                 cam_param['princpt'][1]
                joints_coord_img = self.joint3DToImg(joints_coord_cam, (fx, fy, fu, fv))
                center_2d = self.get_center(joints_coord_img[:, :2], np.ones_like(joints_coord_img[:, 0]))

                bbox = self.get_bbox(joints_coord_img[:, :2], expansion_factor=1.5)  # 从节点获得紧致的bbx
                bbox = self.process_bbox(bbox, img_shape[1], img_shape[0], expansion_factor=1.0)  # 删除无效的
                if bbox is None:
                    continue
                self.dataset_len += 1
                mano_pose = np.array(ann['mano_param']['pose'], dtype=np.float32)
                mano_shape = np.array(ann['mano_param']['shape'], dtype=np.float32)
                mano_trans = np.array(ann['mano_param']['trans'], dtype=np.float32)
                data = {"img_path": img_path, "img_shape": img_shape, "joints_coord_cam": joints_coord_cam,
                        "joints_coord_img": joints_coord_img,
                        "center_2d": center_2d, "cam_param": cam_param,
                        "mano_pose": mano_pose, "mano_shape": mano_shape, "mano_trans": mano_trans}
            else:
                img_path = osp.join(self.root_dir, img['file_name'])
                root_joint_cam = np.array(ann['root_joint_cam'], dtype=np.float32)
                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann['cam_param'].items()}
                bbox = np.array(ann['bbox'], dtype=np.float32)
                center_2d = [bbox[0], bbox[1]]
                rgb = cv2.imread(img_path.replace('png', 'jpg'))
                # cv2.imwrite('./debug/' + str(self.dataset_len) + '.png', rgb)
                data = {"img_path": img_path, "img_shape": img_shape, "root_joint_cam": root_joint_cam,
                        "center_2d": center_2d, "cam_param": cam_param}
                self.dataset_len += 1
            datalist.append(data)

        return datalist

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # idx = idx + 8460
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape = data['img_path'], data['img_shape']

        if 'RGB' in self.input_modal:
            # rgb = load_img(img_path.replace('png', 'jpg'))
            if self.dataset_version == 'v2':
                rgb = cv2.imread(img_path)
            else:
                rgb = cv2.imread(img_path.replace('png', 'jpg'))
            if not isinstance(rgb, np.ndarray):
                raise IOError("Fail to read %s" % img_path)

        depth = self.read_depth_img(img_path.replace('rgb', 'depth'))

        intrinsics = data['cam_param']
        cam_para = (intrinsics['focal'][0], intrinsics['focal'][1], intrinsics['princpt'][0], intrinsics['princpt'][1])

        if self.phase == 'train' or self.phase == 'test' or self.phase == 'train_all':
            joint_xyz = data['joints_coord_cam'].reshape([21, 3])[HO3D2MANO, :] * 1000
            joint_uvd = self.joint3DToImg(joint_xyz, cam_para)

            if self.center_type == 'refine':
                center_xyz = self.center_xyz[idx]
            else:
                center_xyz = joint_xyz.mean(0)
            gt3Dcrop = joint_xyz - center_xyz
        else:
            joint_xyz = np.ones([32, 3])
            joint_uvd = np.ones([32, 3])
            # mesh_uvd = np.ones([32, 3])#
            gt3Dcrop = np.ones([32, 3])
            if self.center_type == 'refine':
                center_xyz = self.center_xyz[idx]
            else:
                center_xyz = joint_xyz.mean(0)

        center_uvd = self.joint3DToImg(center_xyz, cam_para)
        depth_crop, trans = self.Crop_Image_deep_pp(copy.deepcopy(depth), center_uvd, self.cube_size,
                                                    (self.img_size, self.img_size), cam_para)
        if 'RGB' in self.input_modal:
            rgb_crop, trans_rgb = self.Crop_Image_deep_pp_RGB(copy.deepcopy(rgb), center_uvd, self.cube_size,
                                                              (self.img_size, self.img_size), cam_para)

        if 'train' in self.phase:
            mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1],
                                                   rot_range=self.aug_para[2])  # 10, 0.1, 180
            imgD, _, curLabel, cube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, self.cube_size,
                                                                    trans, mode, off, rot, sc, cam_para)
            if 'RGB' in self.input_modal:
                imgRGB, _, curLabel_rgb, cube_rgb, com2D_rgb, M_rgb, _ = self.augmentCrop_RGB(rgb_crop, gt3Dcrop,
                                                                                              center_uvd,
                                                                                              self.cube_size, trans_rgb,
                                                                                              mode, off, rot, sc,
                                                                                              cam_para)
                if self.color_factor != 0:
                    # RGB augment
                    c_up = 1.0 + self.color_factor
                    c_low = 1.0 - self.color_factor
                    color_scale = np.array(
                        [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
                    imgRGB = np.clip(imgRGB * color_scale[None, None, :], 0, 255)

                # RGB normalization
                # imgRGB = self.set_rgb_bg(imgRGB,imgD)
                imgRGB = self.transform(imgRGB.astype(np.float32)) / 255.
            curLabel = curLabel / (cube[2] / 2.0)
            com3D = self.jointImgTo3D(com2D, cam_para)
            mano_pose, mano_shape, mano_trans = data['mano_pose'], data['mano_shape'], data['mano_trans']
            mano_para = np.concatenate((mano_pose, mano_shape, mano_trans), axis=0)
            if mode == 0:
                rot_aug_mat = np.array([[np.cos(np.deg2rad(rot)), -np.sin(np.deg2rad(rot)), 0],
                                        [np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot)), 0],
                                        [0, 0, 1]], dtype=np.float32)
                # mesh_uvd = np.dot(rot_aug_mat, (mesh_uvd - com2D).transpose(1, 0)).transpose(1, 0) + com2D#
        elif self.phase == 'test':
            imgD = self.normalize_img(depth_crop.max(), depth_crop, center_xyz, self.cube_size)
            if 'RGB' in self.input_modal:
                # imgRGB = self.set_rgb_bg(imgRGB, imgD)
                imgRGB = self.transform(rgb_crop.astype(np.float32)) / 255.
            curLabel = gt3Dcrop / (self.cube_size[2] / 2.0)
            cube = np.array(self.cube_size)
            com2D = center_uvd
            M = trans
            com3D = self.jointImgTo3D(com2D, cam_para)
            mano_pose, mano_shape, mano_trans = data['mano_pose'], data['mano_shape'], data['mano_trans']
            mano_para = np.concatenate((mano_pose, mano_shape, mano_trans), axis=0)
        else:
            imgD = self.normalize_img(depth_crop.max(), depth_crop, center_xyz, self.cube_size)
            if 'RGB' in self.input_modal:
                # imgRGB = self.set_rgb_bg(imgRGB, imgD)
                imgRGB = self.transform(rgb_crop.astype(np.float32)) / 255.
            curLabel = gt3Dcrop / (self.cube_size[2] / 2.0)
            cube = np.array(self.cube_size)
            com2D = center_uvd
            M = trans
            com3D = self.jointImgTo3D(com2D, cam_para)
            mano_para = np.ones([55])

        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (cube[0] / 2.0) + com3D, cam_para), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size / 2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (cube[0] / 2.0)

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

        if 'RGB' in self.input_modal:
            data_rgb = imgRGB
        else:
            data_rgb = None

        joint_img = torch.from_numpy(joint_img).float()
        joint = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(cube).float()
        cam_para = torch.from_numpy(np.array(cam_para)).float()

        return data_rgb, data_depth, pcl_sample, joint, joint_img, center, M, cube, cam_para

    def get_center(self, joint_img, joint_valid):
        x_img, y_img = joint_img[:, 0], joint_img[:, 1]
        x_img = x_img[joint_valid == 1]
        y_img = y_img[joint_valid == 1]
        xmin = min(x_img)
        ymin = min(y_img)
        xmax = max(x_img)
        ymax = max(y_img)

        x_center = (xmin + xmax) / 2.
        y_center = (ymin + ymax) / 2.

        return [x_center, y_center]

    def get_bbox(self, joint_img, expansion_factor=1.0):

        x_img, y_img = joint_img[:, 0], joint_img[:, 1]
        xmin = min(x_img)
        ymin = min(y_img)
        xmax = max(x_img)
        ymax = max(y_img)

        x_center = (xmin + xmax) / 2.
        width = (xmax - xmin) * expansion_factor
        xmin = x_center - 0.5 * width
        xmax = x_center + 0.5 * width

        y_center = (ymin + ymax) / 2.
        height = (ymax - ymin) * expansion_factor
        ymin = y_center - 0.5 * height
        ymax = y_center + 0.5 * height

        bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
        return bbox

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

    def read_depth_img(self, depth_filename):
        """Read the depth image in dataset and decode it"""
        depth_scale = 0.00012498664727900177
        depth_img = cv2.imread(depth_filename)
        dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
        dpt = dpt * depth_scale * 1000
        return dpt

    def read_seg_img(self, filename):
        """Read the depth image in dataset and decode it"""
        seg_img = cv2.imread(filename)
        h, w, c = seg_img.shape
        seg_img = seg_img.reshape([-1, 3])
        seg_label = np.zeros([h, w]).reshape([-1])
        seg_label[seg_img[:, 0] == 255] = 1
        seg_label[seg_img[:, 1] == 255] = 2
        seg_label = seg_label.reshape([h, w])
        seg_label = cv2.resize(seg_label, [640, 480], interpolation=cv2.INTER_NEAREST)
        return seg_label


class nyu_loader(loader):
    def __init__(self, root_dir, phase, aug_para=[10, 0.1, 180], img_size=128,
                 cube_size=[250, 250, 250], center_type='refine', joint_num=23, loader=nyu_reader, color_factor=0):
        super(nyu_loader, self).__init__(root_dir, phase, img_size, center_type, 'nyu')
        self.paras = (588.03, 587.07, 320., 240.)
        self.cube_size = np.array(cube_size)
        self.allJoints = True
        self.flip = 1
        self.color_factor = color_factor

        self.croppedSz = img_size
        self.joint_num = joint_num
        self.aug_modes = ['rot', 'com', 'sc', 'none']  # 'rot','com','sc','none'
        self.aug_para = aug_para
        self.transform = transforms.ToTensor()

        data_path = '{}/{}'.format(self.root_dir, self.phase)
        label_path = '{}/joint_data.mat'.format(data_path)
        print('loading data...')
        self.labels = sio.loadmat(label_path)
        self.data_path = data_path

        self.all_joints_uvd = self.labels['joint_uvd'][0][:, joint_select, :][:, calculate, :]
        self.all_joints_xyz = self.labels['joint_xyz'][0][:, joint_select, :][:, calculate, :]
        self.all_joints_xyz = self.jointImgTo3D(self.joint3DToImg(self.all_joints_xyz, flip=-1))

        print('finish!!')
        if center_type == 'refine':
            center_path = '{}/center_{}_refined.txt'.format(data_path, self.phase)
            self.center_xyz = np.loadtxt(center_path)
            self.center_xyz = self.jointImgTo3D(self.joint3DToImg(self.center_xyz, flip=-1))
        elif center_type == 'joint':
            self.center_xyz = self.all_joints_xyz[:, 20, :]
        elif center_type == 'joint_mean':
            self.center_xyz = self.all_joints_xyz.mean(1)
        elif center_type == 'random':
            self.center_xyz = self.all_joints_xyz.mean(1)

        self.loader = loader
        self.test_cubesize = np.ones([8252, 3]) * self.cube_size
        self.test_cubesize[2440:, :] = self.test_cubesize[2440:, :] * 5.0 / 6.0
        self.all_model_para = self.read_modelPara(root_dir, 0)

        self.joint_kernel = torch.Tensor([0.15, 0.15, 0.15, 0.25,
                                          0.15, 0.15, 0.15, 0.25,
                                          0.15, 0.15, 0.15, 0.25,
                                          0.15, 0.15, 0.15, 0.25,
                                          0.25, 0.25, 0.25, 0.35,
                                          0.4, 0.4, 0.4]) * 1.1

    def __getitem__(self, index):
        img_path = self.data_path + '/depth_1_{:07d}.png'.format(index + 1)
        if not os.path.exists(img_path):
            print(img_path)
        depth = self.loader(img_path)
        RGB = cv2.imread(img_path.replace("depth", "rgb"))
        if not isinstance(RGB, np.ndarray):
            print("RGB read fail")
            RGB = np.zeros(shape=(480, 640, 3), dtype=np.uint8)
        joint_xyz = self.all_joints_xyz[index].copy()

        if self.phase == 'test':
            cube_size = self.test_cubesize[index]
        else:
            cube_size = self.cube_size

        center_xyz = self.center_xyz[index]
        center_uvd = self.joint3DToImg(center_xyz)

        gt3Dcrop = joint_xyz - center_xyz.reshape(1, 3)
        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, (self.img_size, self.img_size),
                                                    self.paras)
        rgb_crop, trans_RGB = self.Crop_Image_deep_pp_RGB(RGB, center_uvd, cube_size, (self.img_size, self.img_size),
                                                          self.paras)
        if self.phase == 'train':
            mode, off, rot, sc = self.rand_augment(sigma_com=self.aug_para[0], sigma_sc=self.aug_para[1],
                                                   rot_range=self.aug_para[2])
            imgD, _, curLabel, cube, com2D, M, _ = self.augmentCrop(depth_crop, gt3Dcrop, center_uvd, self.cube_size,
                                                                    trans, mode, off, rot, sc, self.paras)
            imgRGB, _, _, _, _, _, _ = self.augmentCrop_RGB(rgb_crop, gt3Dcrop, center_uvd, self.cube_size,
                                                            trans_RGB, mode, off, rot, sc, self.paras)

            if self.color_factor != 0:
                # RGB augment
                c_up = 1.0 + self.color_factor
                c_low = 1.0 - self.color_factor
                color_scale = np.array(
                    [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
                imgRGB = np.clip(imgRGB * color_scale[None, None, :], 0, 255)
            # RGB normalization
            imgRGB = self.transform(imgRGB.astype(np.float32)) / 255.

            curLabel = curLabel / (cube[2] / 2.0)

        else:
            imgD = self.normalize_img(depth_crop.max(), depth_crop, center_xyz, cube_size)
            imgRGB = self.transform(rgb_crop.astype(np.float32)) / 255.
            curLabel = gt3Dcrop / (cube_size[2] / 2.0)
            cube = np.array(cube_size)
            com2D = center_uvd
            M = trans

        com3D = self.jointImgTo3D(com2D)
        joint_img = transformPoints2D(self.joint3DToImg(curLabel * (cube[0] / 2.0) + com3D), M)
        joint_img[:, 0:2] = joint_img[:, 0:2] / (self.img_size / 2) - 1
        joint_img[:, 2] = (joint_img[:, 2] - com3D[2]) / (cube[0] / 2.0)
        # get pcl
        pcl = self.getpcl(imgD, com3D, cube, M)
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

        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)
        data_rgb = imgRGB
        pcl_sample = torch.from_numpy(pcl_sample).float()
        joint_img = torch.from_numpy(joint_img).float()
        joint = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(cube).float()
        cam_para = torch.from_numpy(np.array(self.paras)).float()

        return data_rgb, data, pcl_sample, joint, joint_img, center, M, cube, cam_para

    def __len__(self):
        return len(self.all_joints_xyz)


def xyz2error(output, joint, center, cube_size):
    output = output.detach().cpu().numpy()
    joint = joint.detach().cpu().numpy()
    center = center.detach().cpu().numpy()
    cube_size = cube_size.detach().cpu().numpy()
    batchsize, joint_num, _ = output.shape
    center = np.tile(center.reshape(batchsize, 1, -1), [1, joint_num, 1])
    cube_size = np.tile(cube_size.reshape(batchsize, 1, -1), [1, joint_num, 1])

    joint_xyz = output * cube_size / 2 + center
    joint_world_select = joint * cube_size / 2 + center

    errors = (joint_xyz - joint_world_select) * (joint_xyz - joint_world_select)
    errors = np.sqrt(np.sum(errors, axis=2))
    return errors


def Matr(theta):
    B = theta.size(0)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    zero_theta = torch.zeros_like(theta)
    one_theta = torch.ones_like(theta)
    row_1 = torch.stack([cos_theta, -sin_theta, zero_theta], dim=-1).view(B, 1, 3)
    row_2 = torch.stack([sin_theta, cos_theta, zero_theta], dim=-1).view(B, 1, 3)
    row_3 = torch.stack([zero_theta, zero_theta, one_theta], dim=-1).view(B, 1, 3)

    return torch.cat([row_1, row_2, row_3], dim=1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
