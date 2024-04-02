import math
import pickle
import random

import torch

from dataloader.loader import loader, transformPoints2D
from dataloader.preprocess import preprocessSTB, uvd2xyz, augmentation, \
    cv2pil, STB_xyz2uvd, read_depth_img
import scipy.io
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import time
from config import opt
import torchvision.transforms as standard

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class STB(loader):
    def __init__(self, setup, split, root_dir, img_size=128, aug_para=[10, 0.2, 180], input_modal='RGBD'):
        super(STB, self).__init__(root_dir, split, img_size, 'joint_mean', 'STB')
        # chose if you want to create a binary for training or evaluation set
        start = time.time()
        self.root = root_dir + '/STB'
        self.mode = split
        self.root_joint_idx = 0
        self.input_modal = input_modal

        self.aug_para = aug_para
        self.cube_size = [200, 200, 200]
        self.aug_modes = ['rot', 'com', 'sc', 'none']  # 'rot', 'com', 'sc', 'none'
        self.flip = 1
        assert self.mode in ['train', 'val', 'test'], 'mode error'
        if self.mode == 'train':
            sequences = ['B2Counting', 'B2Random',
                         'B3Counting', 'B3Random',
                         'B4Counting', 'B4Random',
                         'B5Counting', 'B5Random',
                         'B6Counting', 'B6Random']
        else:
            sequences = ['B1Counting']
        self.image_paths = []
        self.image_paths_d = []
        self.kp_coord_xyz = []
        self.transform = standard.Compose([standard.ToTensor(), standard.Normalize(*mean_std)])
        for seq in sequences:
            matfile = '%s/labels/%s_SK.mat' % (self.root, seq)
            data = scipy.io.loadmat(matfile)
            handPara = data['handPara']
            for id in range(1500):
                image_path = self.root+ '/'
                img_path_rgb = '%s%s/%s_%s_%d.png' % (image_path, seq, 'SK', 'color', id)
                img_path_d = '%s%s/%s_%s_%d.png' % (image_path, seq, 'SK', 'depth', id)
                self.image_paths.append(img_path_rgb)
                self.image_paths_d.append(img_path_d)
                kp_coord_xyz = handPara[:, :, id]
                self.kp_coord_xyz.append(kp_coord_xyz)
        print('It takes %f seconds to read %d images in STB dataset!' %(time.time()-start, len(self.image_paths)))

    def __getitem__(self, idx):
        # camera intrinsic parameters
        R = np.array([0.00531, -0.01196, 0.00301])
        T = np.array([-24.0381, -0.4563, -1.2326])
        fx = 607.92271
        fy = 607.88192
        tx = 314.78337
        ty = 236.42484
        cam_para = (fx, fy, tx, ty)
        K = np.array([[fx, 0, tx], [0, fy, ty], [0, 0, 1]])
        rotationMatrix = cv2.Rodrigues(R)[0]
        T = np.reshape(T, [3, 1])

        # image path
        image = cv2.imread(self.image_paths[idx])
        img_d = read_depth_img(self.image_paths_d[idx])
        # read annotation
        xyz = self.kp_coord_xyz[idx].transpose(1, 0)
        uvd = STB_xyz2uvd(xyz, K, rotationMatrix, T)
        # preprocessing data
        uvd, crop_center, crop_size = preprocessSTB(uvd)
        xyz = uvd2xyz(uvd, K)

        center_xyz = xyz.mean(0)
        gt3Dcrop = xyz - center_xyz
        center_uvd = self.joint3DToImg(center_xyz, cam_para)

        depth_crop, trans = self.Crop_Image_deep_pp(img_d, center_uvd, self.cube_size, (self.img_size, self.img_size),cam_para)

        if 'RGB' in self.input_modal:
            rgb_crop, trans_rgb = self.Crop_Image_deep_pp_RGB(image, center_uvd, self.cube_size,
                                                              (self.img_size, self.img_size), cam_para)
        if self.mode == 'train':
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
                color_factor = 0.2
                if color_factor != 0:
                    # RGB数据增强
                    c_up = 1.0 + color_factor
                    c_low = 1.0 - color_factor
                    color_scale = np.array(
                        [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
                    imgRGB = np.clip(imgRGB * color_scale[None, None, :], 0, 255)

                # RGB图像归一化
                # imgRGB = self.set_rgb_bg(imgRGB,imgD)
                imgRGB = self.transform(imgRGB.astype(np.float32)) / 255.

            curLabel = curLabel / (cube[2] / 2.0)
            if mode == 0:
                rot_aug_mat = np.array([[np.cos(np.deg2rad(rot)), -np.sin(np.deg2rad(rot)), 0],
                                        [np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot)), 0],
                                        [0, 0, 1]], dtype=np.float32)

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
            # print(data_rgb.shape)
        else:
            data_rgb = None

        joint_img = torch.from_numpy(joint_img).float()
        joint = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(cube).float()

        cam_para = torch.from_numpy(np.array(cam_para)).float()

        return data_rgb, data_depth, pcl_sample, joint, joint_img, center, M, cube, cam_para

    def __len__(self):
        return len(self.image_paths)

