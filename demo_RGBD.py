import math
import os
import sys
from torch.nn.parallel.data_parallel import DataParallel
# sys.path.append("../KeypointFusion/")
from util import vis_tool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import KPFusion
from config import opt
import cv2
import numpy as np
import torch.nn
import random
import torch.backends.cudnn as cudnn
from util.generateFeature import GFM
from dataloader.processing_depth import HO3D
import torchvision.transforms as transforms

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


class Model_RGBD(object):
    def __init__(self, opt, cam_para):
        self.config = opt
        self.img_size = self.config.input_size
        # self.cam_para = (906.96, 906.79, 956.75, 547.23)
        self.cam_para = cam_para
        self.sample_num = 1024
        self.flip = 1
        self.cube = opt.cube_size

        self.data_rt = self.config.root_dir + "/" + self.config.dataset
        self.depthloader = HO3D()

        self.transform = transforms.ToTensor()

        cudnn.benchmark = False
        self.dataset = 'nyu_all' if 'nyu' in self.config.dataset else 'hands'
        self.joint_num = 23 if 'nyu' in self.config.dataset else self.config.joint_num

        self.net = KPFusion(self.config.net, self.config.pretrain, self.joint_num, self.dataset,
                            './MANO/', kernel_size=self.config.feature_para[0])

        self.net = DataParallel(self.net).cuda()
        # self.net = self.net.cuda()
        self.GFM_ = GFM()
        self.start_epoch = 0

        # load model
        if self.config.load_model != '':
            print('loading model from %s' % self.config.load_model)
            checkpoint = torch.load(self.config.load_model, map_location=lambda storage, loc: storage)
            checkpoint_model = checkpoint['model']
            model_dict = self.net.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint_model.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.net.load_state_dict(model_dict)
            print('RGBD model loading finished')

    def estimate_pose_RGBD(self, inputs):

        self.phase = 'evaluation'
        self.net.eval()
        meta_info_file_list = []
        outputs_file_list = []

        for input in inputs:
            img_rgb, img_d, bbox, hand_type = input['rgb'], input['depth'], input['bbox'], input['hand_type']

            MANO2HO3D = [0,
                         1, 2, 3,
                         4, 5, 6,
                         7, 8, 9,
                         10, 11, 12,
                         13, 14, 15,
                         20, 16, 17, 19, 18]

            center = self.get_center_from_bbx(img_d, bbox)

            crop_RGB, _ = self.Crop_Image_deep_pp_RGB(img_rgb, center, self.cube, (self.img_size, self.img_size),
                                                      self.cam_para)
            crop_RGB = self.transform(crop_RGB.astype(np.float32)) / 255.

            img_rgb = crop_RGB.unsqueeze(0).float()

            img, pcl, com3D, M, cube = self.process_depth(self.cube, img_d, center)  # trans=M
            crop_img_d = (img.permute(1, 2, 0) + 1) / 2 * 255.0
            crop_img_d = cv2.cvtColor(crop_img_d.numpy().astype(np.uint8), cv2.COLOR_GRAY2RGB)

            crop_img_rgb = (img_rgb[0] * 255).permute(1, 2, 0)
            crop_img_rgb = cv2.cvtColor(crop_img_rgb.numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)

            img = img.unsqueeze(0).float()
            pcl = pcl.unsqueeze(0).float()

            img_rgb, img, pcl, cam_para = img_rgb.cuda(), img.cuda(), pcl.cuda(), torch.tensor(self.cam_para).cuda()

            center, M, cube = torch.tensor(com3D).cuda(), M.cuda(), cube.cuda()

            center = center.unsqueeze(0).float()

            cube = cube.unsqueeze(0).float()
            M = M.unsqueeze(0).float()
            cam_para = cam_para.unsqueeze(0).float()

            results, spatial_weight, _ = self.net(img_rgb, img, pcl, self.depthloader, center, M, cube, cam_para, 0.8)

            joint_xyz_list = []
            joint_uvd_list = []
            for index, stage_type in enumerate(self.config.stage_type):
                if stage_type == 1:
                    pixel_pd = results[index]
                    joint_uvd = self.GFM_.feature2joint(img, pixel_pd, self.config.feature_type,
                                                        self.config.feature_para)
                    joint_xyz = self.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para)

                elif stage_type == 2 or stage_type == 3:
                    joint_xyz = results[index]
                    joint_uvd = self.xyz_nl2uvdnl_tensor(joint_xyz, com3D, M, cube, cam_para)
                    # print((joint_uvd + 1) / 2 * 128)

                # joint_xyz_world = joint_xyz * cube.unsqueeze(1) / 2 + center.unsqueeze(1)
                # joint_xyz_world = joint_xyz_world.detach().cpu()

                joint_xyz_list.append(joint_xyz)
                joint_uvd_list.append(joint_uvd)

            joint_xyz_world = joint_xyz_list[-1] * cube.unsqueeze(1) / 2 + center.unsqueeze(1)

            joint_xyz_world = joint_xyz_world.detach().cpu().numpy()[:, MANO2HO3D, :]
            joint_xyz_world *= np.array([1, -1, -1]) / 1000

            # uvd2img
            # joint_uvd = joint_uvd_list[0].squeeze(0)
            # joint_uvd = joint_uvd_list[-1].squeeze(0)
            joint_uvd = joint_uvd_list[2].squeeze(0)
            joint_uvd[:, 2] = (joint_uvd[:, 2] * cube[0][0] / 2) + center[0][2]

            # back to original image
            coord_uvd_full = joint_uvd.clone()

            coord_uvd_full = self.transformPoints2D(coord_uvd_full.cpu().data.numpy(),
                                                    np.linalg.inv(M.cpu().data.numpy()))
            img_show = vis_tool.draw_2d_pose(img[0], joint_uvd, 'ho3d')

            rgb_show = vis_tool.draw_2d_pose_rgb(img_rgb[0], joint_uvd, 'ho3d')
            rgb_show = cv2.cvtColor(rgb_show, cv2.COLOR_BGR2RGB)

            meta_info = {
                'crop_img': crop_img_rgb,
                'crop_img_d': crop_img_d,
                'K': None,
                'img2bb_trans': M,
                'bb2img_trans': torch.linalg.inv(M),
                'scale': None,
                'center': center,
                'pose_rgb': rgb_show,
                'pose_d': img_show,
            }
            outputs = {
                'mesh_uvd': None,
                'mesh_xyz': None,
                'joint_uvd': coord_uvd_full,
                'joint_xyz': joint_xyz_world[0]
            }
            meta_info_file_list.append(meta_info)
            outputs_file_list.append(outputs)
        return meta_info_file_list, outputs_file_list

    def uvd_nl2xyznl_tensor(self, uvd, center, m, cube, cam_paras):
        batch_size, point_num, _ = uvd.size()
        device = uvd.device
        cube_size_t = cube.to(device).view(batch_size, 1, 3).repeat(1, point_num, 1)
        center_t = center.to(device).view(batch_size, 1, 3).repeat(1, point_num, 1)
        M_t = m.to(device).view(batch_size, 1, 3, 3)
        M_inverse = torch.linalg.inv(M_t).repeat(1, point_num, 1, 1)

        uv_unnormal = (uvd[:, :, 0:2] + 1) * (self.config.input_size / 2)
        d_unnormal = (uvd[:, :, 2:]) * (cube_size_t[:, :, 2:] / 2.0) + center_t[:, :, 2:]
        uvd_unnormal = torch.cat((uv_unnormal, d_unnormal), dim=-1)
        uvd_world = self.get_trans_points(uvd_unnormal, M_inverse)
        xyz = self.pointsImgTo3D(uvd_world, cam_paras)
        # xyz_noraml = (xyz - center_t) / (cube_size_t / 2.0)
        # return xyz_noraml
        return xyz

    def transformPoints2D(self, pts, M):
        """
        Transform points in 2D coordinates
        :param pts: point coordinates
        :param M: transformation matrix
        :return: transformed points
        """
        ret = pts.copy()
        for i in range(pts.shape[0]):
            ret[i, 0:2] = self.transformPoint2D(pts[i, 0:2], M)
        return ret

    def transformPoint2D(self, pt, M):
        """
        Transform point in 2D coordinates
        :param pt: point coordinates
        :param M: transformation matrix
        :return: transformed point
        """
        pt2 = np.dot(np.asarray(M).reshape((3, 3)), np.asarray([pt[0], pt[1], 1]))
        return np.asarray([pt2[0] / pt2[2], pt2[1] / pt2[2]])

    def pointsImgTo3D(self, point_uvd, paras, flip=None):
        # if flip == None:
        #     flip = self.flip
        point_xyz = torch.zeros_like(point_uvd).to(point_uvd.device)
        point_xyz[:, :, 0] = (point_uvd[:, :, 0] - paras[:, 2].unsqueeze(1)) * point_uvd[:, :, 2] / paras[:,
                                                                                                    0].unsqueeze(1)
        point_xyz[:, :, 1] = self.flip * (point_uvd[:, :, 1] - paras[:, 3].unsqueeze(1)) * point_uvd[:, :, 2] / paras[:,
                                                                                                                1].unsqueeze(
            1)
        point_xyz[:, :, 2] = point_uvd[:, :, 2]
        return point_xyz

    def get_trans_points(self, joints, M):
        device = joints.device
        joints_mat = torch.cat((joints[:, :, 0:2], torch.ones(joints.size(0), joints.size(1), 1).to(device)), dim=-1)
        joints_trans_xy = torch.matmul(M, joints_mat.unsqueeze(-1)).squeeze(-1)[:, :, 0:2]
        joints_trans_z = joints[:, :, 2:]
        return torch.cat((joints_trans_xy, joints_trans_z), dim=-1)

    def get_center(self, img, xywh, upper=650, lower=1):
        centers = np.array([0.0, 0.0, 300.0])
        count = 0

        x_lower = int(xywh[0])
        x_upper = int(xywh[0] + xywh[2])
        y_lower = int(xywh[1])
        y_upper = int(xywh[1] + xywh[3])

        for y in range(y_lower, y_upper):
            for x in range(x_lower, x_upper):
                if img[y, x] <= upper and img[y, x] >= lower:
                    centers[0] += x
                    centers[1] += y
                    centers[2] += img[y, x]
                    count += 1
        if count:
            centers /= count
        return centers

    def get_center_from_bbx(self, depth, bbx, upper=1500, lower=171):
        centers = np.array([0.0, 0.0, 300.0])
        x_min = int(bbx[0])
        x_max = int(bbx[0] + bbx[2])
        y_min = int(bbx[1])
        y_max = int(bbx[1] + bbx[3])

        img = depth[int(y_min):int(y_max), int(x_min):int(x_max)]  # crop
        flag = np.logical_and(img <= upper, img >= lower)
        x = np.linspace(0, img.shape[1], img.shape[1])
        y = np.linspace(0, img.shape[0], img.shape[0])
        xv, yv = np.meshgrid(x, y)
        centers[0] = np.mean(xv[flag])
        centers[1] = np.mean(yv[flag])
        centers[2] = np.mean(img[flag])
        if centers[2] <= 0:
            centers[2] = 300.0
        if not flag.any():
            centers[0] = 0
            centers[1] = 0
            centers[2] = 300.0
        centers[0] += bbx[0]  # coordinate in the original image
        centers[1] += bbx[1]
        return centers

    def xyz_nl2uvdnl_tensor(self, joint_xyz, center, M, cube_size, cam_paras):
        device = joint_xyz.device
        batch_size, joint_num, _ = joint_xyz.size()
        cube_size_t = cube_size.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
        center_t = center.to(device).view(batch_size, 1, 3).repeat(1, joint_num, 1)
        M_t = M.to(device).view(batch_size, 1, 3, 3).repeat(1, joint_num, 1, 1)

        joint_temp = joint_xyz * cube_size_t / 2.0 + center_t
        joint_uvd = self.points3DToImg(joint_temp, cam_paras)
        joint_uvd = self.get_trans_points(joint_uvd, M_t)
        joint_uv = joint_uvd[:, :, 0:2] / self.config.input_size * 2.0 - 1
        joint_d = (joint_uvd[:, :, 2:] - center_t[:, :, 2:]) / (cube_size_t[:, :, 2:] / 2)
        joint = torch.cat((joint_uv, joint_d), dim=-1)
        return joint

    def points3DToImg(self, joint_xyz, para, flip=None):

        joint_uvd = torch.zeros_like(joint_xyz).to(joint_xyz.device)
        joint_uvd[:, :, 0] = (
                joint_xyz[:, :, 0] * para[:, 0].unsqueeze(1) / (joint_xyz[:, :, 2] + 1e-8) + para[:, 2].unsqueeze(
            1))
        joint_uvd[:, :, 1] = (self.flip * joint_xyz[:, :, 1] * para[:, 1].unsqueeze(1) / (joint_xyz[:, :, 2]) + para[:,
                                                                                                                3].unsqueeze(
            1))
        joint_uvd[:, :, 2] = joint_xyz[:, :, 2]
        return joint_uvd

    def process_depth(self, cube_size, depth, center):

        center_uvd = center
        depth_crop, trans = self.Crop_Image_deep_pp(depth, center_uvd, cube_size, (self.img_size, self.img_size),
                                                    self.cam_para)
        imgD = self.normalize_img(depth_crop.max(), depth_crop, center, cube_size)
        cube = np.array(cube_size)
        com2D = center_uvd
        M = trans

        com3D = self.jointImgTo3D(com2D)

        # get pcl
        pcl = self.getpcl(imgD, com3D, cube, M, self.cam_para)
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

        data = torch.from_numpy(imgD).float()
        data = data.unsqueeze(0)

        # joint_img = torch.from_numpy(joint_img).float()
        # joint = torch.from_numpy(curLabel).float()
        center = torch.from_numpy(com3D).float()
        M = torch.from_numpy(M).float()
        cube = torch.from_numpy(cube).float()

        return data, pcl_sample, center, M, cube

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

    def normalize_img(self, premax, imgD, com, cube):
        imgD[imgD == premax] = com[2] + (cube[2] / 2.)
        imgD[imgD == 0] = com[2] + (cube[2] / 2.)
        imgD[imgD >= com[2] + (cube[2] / 2.)] = com[2] + (cube[2] / 2.)
        imgD[imgD <= com[2] - (cube[2] / 2.)] = com[2] - (cube[2] / 2.)
        imgD -= com[2]
        imgD /= (cube[2] / 2.)
        return imgD

    def jointImgTo3D(self, uvd, paras=None, flip=None):
        if isinstance(paras, tuple):
            fx, fy, fu, fv = paras
        else:
            fx, fy, fu, fv = self.cam_para
        if flip == None:
            flip = 1
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
            sz = (dsize[0], int(hb * dsize[0] / wb))  #
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

    def comToBounds(self, com, size, paras):
        fx, fy, fu, fv = paras
        zstart = com[2] - size[2] / 2.
        zend = com[2] + size[2] / 2.

        xstart = int(np.floor((com[0] * com[2] / fx - size[0] / 2.) / com[2] * fx + 0.5))
        xend = int(np.floor((com[0] * com[2] / fx + size[0] / 2.) / com[2] * fx + 0.5))
        ystart = int(np.floor((com[1] * com[2] / fy - size[1] / 2.) / com[2] * fy + 0.5))
        yend = int(np.floor((com[1] * com[2] / fy + size[1] / 2.) / com[2] * fy + 0.5))

        return xstart, xend, ystart, yend, zstart, zend

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


if __name__ == '__main__':
    # set_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    rgb = cv2.imread('./visualization/box.png')
    depth = cv2.imread('./visualization/box_d.png', cv2.IMREAD_ANYDEPTH)

    bbox = [885, 515.50, 178.00, 127.00] # xywh
    bbox[0] -= bbox[2] / 2
    bbox[1] -= bbox[3] / 2

    input = [{'rgb': rgb, 'depth': depth, 'bbox': bbox, 'hand_type': 'right'}]

    # set the internal parameters of your RGB-D camera
    model_RGBD = Model_RGBD(opt, cam_para=(906.96, 906.79, 956.75, 547.23))
    meta_info, result = model_RGBD.estimate_pose_RGBD(input)

    cv2.imwrite('./visualization/box_cropRGB.png', cv2.cvtColor(np.uint8(meta_info[0]['crop_img']), cv2.COLOR_BGR2RGB))
    cv2.imwrite('./visualization/box_cropD.png', meta_info[0]['crop_img_d'])

    cv2.imwrite('./visualization/box_poseRGB.png', cv2.cvtColor(np.uint8(meta_info[0]['pose_rgb']), cv2.COLOR_BGR2RGB))
    cv2.imwrite('./visualization/box_poseD.png', meta_info[0]['pose_d'])
