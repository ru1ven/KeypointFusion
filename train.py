import os
import time

import torchvision
from torch.nn.parallel.data_parallel import DataParallel

from dataloader.dataset_STB import STB
from model.model import KPFusion
from config import opt
from util.eval_utils import eval_auc

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
import cv2
import shutil
import logging
import numpy as np

import torch.nn
from tqdm import tqdm
import random

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR

from tensorboardX import SummaryWriter

from dataloader import loader
from util.generateFeature import GFM

from model.loss import SmoothL1Loss
from util import vis_tool

import json

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.data_rt = self.config.root_dir + "/" + self.config.dataset
        if self.config.model_save == '':
            self.model_save = self.config.net + '_ips' + str(
                self.config.input_size)

            self.model_dir = './checkpoint/' + self.config.dataset + '/' + self.model_save

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.model_dir + '/img')
            os.makedirs(self.model_dir + '/debug')
            os.makedirs(self.model_dir + '/files')

        # save core file
        shutil.copyfile('train.py', self.model_dir + '/files/train.py')
        shutil.copyfile('./config.py', self.model_dir + '/files/config.py')
        shutil.copyfile('model/model.py', self.model_dir + '/files/model.py')
        #shutil.copyfile('./model/fusion_layer.py', self.model_dir + '/files/fusion_layer.py')
        shutil.copyfile('./dataloader/loader.py', self.model_dir + '/files/loader.py')

        # save config
        with open(self.model_dir + '/config.json', 'w') as f:
            for k, v in self.config.__class__.__dict__.items():
                if not k.startswith('_'):
                    print(str(k) + ":" + str(v))
                    f.writelines(str(k) + ":" + str(v) + '\n')

        cudnn.benchmark = False
        self.dataset = 'nyu_all' if 'nyu' in self.config.dataset else 'hands'
        self.joint_num = 23 if 'nyu' in self.config.dataset else self.config.joint_num

        self.net = KPFusion(self.config.net, self.config.pretrain, self.joint_num, self.dataset,
                            './MANO/', kernel_size=self.config.feature_para[0])

        self.net = DataParallel(self.net).cuda()
        self.GFM_ = GFM()

        optimList = [{"params": self.net.parameters(), "initial_lr": self.config.lr}]
        # init optimizer
        if self.config.opt == 'sgd':
            self.optimizer = SGD(optimList, lr=self.config.lr, momentum=0.9, weight_decay=1e-4)
        elif self.config.opt == 'adam':
            self.optimizer = Adam(optimList, lr=self.config.lr)
        elif self.config.opt == 'adamw':
            self.optimizer = AdamW(optimList, lr=self.config.lr, weight_decay=0.01)

        self.L1Loss = SmoothL1Loss().cuda()
        self.BECLoss = torch.nn.BCEWithLogitsLoss().cuda()

        self.L2Loss = torch.nn.MSELoss().cuda()
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

        # fine-tune model
        if self.config.finetune_dir != '':
            print('loading model from %s' % self.config.finetune_dir)
            checkpoint = torch.load(self.config.finetune_dir, map_location=lambda storage, loc: storage)
            checkpoint_model = checkpoint['model']
            model_dict = self.net.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint_model.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.net.load_state_dict(model_dict)

        # init scheduler
        self.scheduler = StepLR(self.optimizer, step_size=self.config.step_size, gamma=0.1, last_epoch=self.start_epoch)

        if self.config.dataset == 'dexycb':
            if self.config.phase == 'train':
                self.trainData = loader.DexYCBDataset(self.config.dexycb_setup, 'train', self.config.root_dir,
                                                      aug_para=self.config.augment_para,
                                                      img_size=self.config.input_size)
                self.trainLoader = DataLoader(self.trainData, batch_size=self.config.batch_size, shuffle=True,
                                              num_workers=8)
            self.testData = loader.DexYCBDataset(self.config.dexycb_setup, 'test', self.config.root_dir,
                                                 img_size=self.config.input_size)
            self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False,
                                         num_workers=8)

        elif self.config.dataset == 'ho3d':
            if 'train' in self.config.phase:
                self.trainData = loader.HO3D('train_all', self.config.root_dir,
                                             dataset_version=config.ho3d_version,
                                             aug_para=self.config.augment_para,
                                             img_size=self.config.input_size,
                                             cube_size=self.config.cube_size, color_factor=self.config.color_factor,
                                             center_type='joint_mean')
                self.trainLoader = DataLoader(self.trainData, batch_size=self.config.batch_size, shuffle=True,
                                              num_workers=4)
            self.testData = loader.HO3D('test', self.config.root_dir, dataset_version=config.ho3d_version,
                                        img_size=self.config.input_size, cube_size=self.config.cube_size,
                                        center_type='joint_mean', aug_para=[0, 0, 0])
            self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False,
                                         num_workers=4)

            self.evalData = loader.HO3D('eval', self.config.root_dir, dataset_version=config.ho3d_version,
                                        img_size=self.config.input_size, cube_size=self.config.cube_size,

                                        aug_para=[0, 0, 0])
            self.evalLoader = DataLoader(self.evalData, batch_size=self.config.batch_size, shuffle=False,
                                         num_workers=4)

        elif self.config.dataset == 'nyu':
            if 'train' in self.config.phase:
                self.trainData = loader.nyu_loader(self.data_rt, 'train', aug_para=self.config.augment_para,
                                                   img_size=self.config.input_size,
                                                   cube_size=self.config.cube_size,
                                                   center_type=self.config.center_type,
                                                   color_factor=self.config.color_factor)
                self.trainLoader = DataLoader(self.trainData, batch_size=self.config.batch_size, shuffle=True,
                                              num_workers=4)
            self.testData = loader.nyu_loader(self.data_rt, 'test', img_size=self.config.input_size,
                                              cube_size=self.config.cube_size,
                                              center_type=self.config.center_type, aug_para=[0, 0, 0])
            self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False,
                                         num_workers=4)
        elif self.config.dataset == 'STB':
            if self.config.phase == 'train':
                self.trainData = STB(self.config.dexycb_setup, 'train', self.config.root_dir,
                                     aug_para=self.config.augment_para,
                                     img_size=self.config.input_size)
                self.trainLoader = DataLoader(self.trainData, batch_size=self.config.batch_size, shuffle=True,
                                              num_workers=8)
            self.testData = STB(self.config.dexycb_setup, 'test', self.config.root_dir,
                                img_size=self.config.input_size)
            self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False,
                                         num_workers=8)
        else:
            raise NotImplementedError()

        self.test_error = 10000
        self.min_error = 100

        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                            filename=os.path.join(self.model_dir, 'train.log'), level=logging.INFO)
        logging.info('======================================================')
        self.min_error = 100
        self.writer = SummaryWriter('runs/' + self.config.dataset)

    def train(self):
        self.phase = 'train'
        for epoch in range(self.start_epoch, self.config.max_epoch):
            self.net.train()
            for ii, data in tqdm(enumerate(self.trainLoader)):
                joint_xyz_list = []
                joint_uvd_list = []

                img_rgb, img, pcl, xyz_gt, uvd_gt, center, M, cube, cam_para = data
                img_rgb, img, pcl, uvd_gt, xyz_gt, cam_para = img_rgb.cuda(), img.cuda(), pcl.cuda(), uvd_gt.cuda(), xyz_gt.cuda(), cam_para.cuda()
                center, M, cube = center.cuda(), M.cuda(), cube.cuda()

                self.optimizer.zero_grad()
                iter_num = ii + (self.trainData.__len__() // self.config.batch_size) * epoch

                results, spatial_weight, _ = self.net(img_rgb, img, pcl, self.trainData, center, M, cube, cam_para, 0.8)
                loss = 0
                for index, stage_type in enumerate(self.config.stage_type):

                    if stage_type == 1:  # pixel-wise backbone
                        pixel_pd = results[index]  # B x 5J x FS x FS
                        feature_size = pixel_pd.size(-1)
                        pixel_gt = self.GFM_.joint2feature(uvd_gt, img, self.config.feature_para, feature_size,
                                                           self.config.feature_type)
                        joint_uvd = self.GFM_.feature2joint(img, pixel_pd, self.config.feature_type,
                                                            self.config.feature_para)
                        joint_xyz = self.trainData.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para)
                        loss_pixel = self.L1Loss(pixel_pd[:, :pixel_gt.size(1)],
                                                 pixel_gt) * self.config.deconv_weight
                        loss_coord = self.L1Loss(joint_uvd, uvd_gt) * self.config.coord_weight
                        loss += (loss_pixel + loss_coord)

                        joint_xyz_list.append(joint_xyz)
                        joint_uvd_list.append(joint_uvd)
                        batch_joint_error = self.xyz2error(joint_xyz, xyz_gt, center, cube)
                        error = np.mean(batch_joint_error)

                        self.writer.add_scalar('loss_pixel', loss_pixel, global_step=iter_num)
                        self.writer.add_scalar('loss_coord', loss_coord, global_step=iter_num)

                    elif stage_type == 2 or stage_type == 3:  # RGB/D KFAM

                        joint_xyz = results[index]

                        joint_uvd = self.trainData.xyz_nl2uvdnl_tensor(joint_xyz, center, M, cube, cam_para)
                        loss_coord = self.L1Loss(joint_xyz, xyz_gt) * self.config.coord_weight
                        loss += loss_coord

                        joint_xyz_list.append(joint_xyz)
                        joint_uvd_list.append(joint_uvd)
                        batch_joint_error = self.xyz2error(joint_xyz, xyz_gt, center, cube)
                        error = np.mean(batch_joint_error)

                    self.writer.add_scalar('error_{}'.format(index), error, global_step=iter_num)

                # hmloss
                for index, sw in enumerate(spatial_weight):
                    if epoch <= self.config.spatial_epoch[index] and sw is not None:
                        if index == 0:
                            hm_gt = self.GFM_.joint2heatmap(uvd_gt[:, :, :2], self.config.feature_para[0], feature_size,sigma=3)
                        else:
                            hm_gt = self.GFM_.joint2heatmap(uvd_gt[:, :, :2], self.config.feature_para[0], feature_size,sigma=2)

                        spatial_weight_gt = hm_gt / hm_gt.max()

                        loss_spatialweight = self.L1Loss(sw, spatial_weight_gt) * self.config.spatial_weight[index]
                        loss += loss_spatialweight
                        self.writer.add_scalar('loss_spatial_{}'.format(index), loss_spatialweight,global_step=iter_num)

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)
                self.optimizer.step()

            test_error = self.test(epoch)

            if test_error <= self.min_error:
                self.min_error = test_error
                save = {
                    "model": self.net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch
                }
                torch.save(
                    save,
                    self.model_dir + "/best.pth"
                )
            save = {
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch
            }
            # if epoch >=19 and self.config.dataset=='ho3d':
            #     torch.save(
            #         save,
            #         self.model_dir + "/epoch_{}.pth".format(epoch)
            #     )
            torch.save(
                save,
                self.model_dir + "/latest.pth"
            )

            if self.config.scheduler == 'auto':
                self.scheduler.step(test_error)
            elif self.config.scheduler == 'step':
                self.scheduler.step(epoch)
            elif self.config.scheduler == 'multi_step':
                self.scheduler.step()
            else:
                pass

    @torch.no_grad()
    def test(self, epoch=-1):
        self.phase = 'test'
        self.result_file_list = []
        for index in range(len(self.config.stage_type)):
            self.result_file_list.append(open(self.model_dir + '/test_%d.txt' % (index), 'w'))
        self.id_file = open(self.model_dir + '/id.txt', 'w')
        self.mano_file = open(self.model_dir + '/eval_mano.txt', 'w')
        self.net.eval()
        batch_num = 0
        error_list_all_batch = []
        error_list = [0] * len(self.config.stage_type)
        PA_error_list = [0] * len(self.config.stage_type)
        for ii, data in tqdm(enumerate(self.testLoader)):

            img_rgb, img, pcl, xyz_gt, uvd_gt, center, M, cube, cam_para = data
            img_rgb, img, pcl, uvd_gt, xyz_gt, cam_para = img_rgb.cuda(), img.cuda(), pcl.cuda(), uvd_gt.cuda(), xyz_gt.cuda(), cam_para.cuda()

            center, M, cube = center.cuda(), M.cuda(), cube.cuda()

            results, spatial_weight, _ = self.net(img_rgb, img, pcl, self.testData, center, M, cube, cam_para,0.8)

            batch_num += 1
            joint_error_list = []
            PA_joint_error_list = []

            for index, stage_type in enumerate(self.config.stage_type):
                if stage_type == 0:
                    joint_uvd = results[index]
                    joint_xyz = self.testData.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para)
                    joint_errors = self.xyz2error(joint_xyz, xyz_gt, center, cube, self.result_file_list[index])
                    batch_errors = np.mean(joint_errors, axis=-1)
                elif stage_type == 1:

                    pixel_pd = results[index]
                    joint_uvd = self.GFM_.feature2joint(img, pixel_pd, self.config.feature_type,
                                                        self.config.feature_para)
                    joint_xyz = self.testData.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para)
                    joint_errors = self.xyz2error(joint_xyz, xyz_gt, center, cube, self.result_file_list[index])
                    batch_errors = np.mean(joint_errors, axis=-1)

                    joint_errors_aligned = 0
                    for b in range(joint_xyz.shape[0]):

                        j, j_gt = joint_xyz[b].cpu().numpy(), xyz_gt[b].cpu().numpy()
                        if self.dataset != 'STB':
                            joint_xyz_aligned = self.GFM_.rigid_align(j, j_gt)
                        else:
                            joint_xyz_aligned = joint_xyz - (joint_xyz[0] - xyz_gt[0])

                        joint_errors_aligned += self.xyz2error(torch.from_numpy(joint_xyz_aligned).cuda().unsqueeze(0),
                                                               xyz_gt[b].unsqueeze(0), center[b].unsqueeze(0),
                                                               cube[b].unsqueeze(0), self.result_file_list[index])
                    batch_errors_aligned = joint_errors_aligned / joint_xyz.shape[0]

                elif stage_type == 2 or stage_type == 3:
                    joint_xyz = results[index]
                    joint_errors = self.xyz2error(joint_xyz, xyz_gt, center, cube, self.result_file_list[index])
                    batch_errors = np.mean(joint_errors, axis=-1)

                    ##
                    joint_errors_aligned = 0
                    for b in range(joint_xyz.shape[0]):
                        # j, j_gt = np.array(joint_xyz[b].cpu()), np.array(xyz_gt[b].cpu())
                        j, j_gt = joint_xyz[b].cpu().numpy(), xyz_gt[b].cpu().numpy()
                        if self.dataset != 'STB':
                            joint_xyz_aligned = self.GFM_.rigid_align(j, j_gt)
                        else:
                            joint_xyz_aligned = joint_xyz - (joint_xyz[0] - xyz_gt[0])
                        joint_errors_aligned += self.xyz2error(torch.from_numpy(joint_xyz_aligned).cuda().unsqueeze(0),
                                                               xyz_gt[b].unsqueeze(0), center[b].unsqueeze(0),
                                                               cube[b].unsqueeze(0),
                                                               self.result_file_list[index])

                    batch_errors_aligned = joint_errors_aligned / joint_xyz.shape[0]

                joint_error_list.append(joint_errors)
                error = np.mean(batch_errors)
                error_list[index] += error
                joint_xyz_world = joint_xyz * cube.unsqueeze(1) / 2 + center.unsqueeze(1)

                PA_error = np.mean(batch_errors_aligned)
                PA_error_list[index] += PA_error

            error_list_all_batch.append(joint_error_list)
        eval_auc(error_list_all_batch, self.testLoader.__len__())

        error_info = '%d epochs:  ' % epoch
        for index in range(len(error_list)):
            print("[mean_Error %.3f]" % (error_list[index] / batch_num))
            error_info += ' error' + str(index) + ": %.3f" % (error_list[index] / batch_num) + ' '

            print("[PA_mean_Error %.3f]" % (PA_error_list[index] / batch_num))
            error_info += ' error' + str(index) + ": %.3f" % (PA_error_list[index] / batch_num) + ' '
        logging.info(error_info)
        return error_list[-1] / batch_num

    @torch.no_grad()
    def evalution(self, epoch=-1):
        self.phase = 'evaluation'
        self.net.eval()
        joint_list = []
        mesh_list = []
        MANO2HO3D = [0,
                     1, 2, 3,
                     4, 5, 6,
                     7, 8, 9,
                     10, 11, 12,
                     13, 14, 15,
                     20, 16, 17, 19, 18]
        self.result_file_list = []
        for index in range(len(self.config.stage_type)):
            self.result_file_list.append(open(self.model_dir + '/test_%d.txt' % (index), 'w'))
        self.joint_file = open(self.model_dir + '/eval_joint.txt', 'w')
        self.mesh_file = open(self.model_dir + '/eval_mesh.txt', 'w')
        self.mano_file = open(self.model_dir + '/eval_mano.txt', 'w')

        for ii, data in tqdm(enumerate(self.evalLoader)):
            img_rgb, img, pcl, xyz_gt, uvd_gt, center, M, cube, cam_para = data
            img_rgb, img, pcl, uvd_gt, xyz_gt, cam_para = img_rgb.cuda(), img.cuda(), pcl.cuda(), uvd_gt.cuda(), xyz_gt.cuda(), cam_para.cuda()
            center, M, cube = center.cuda(), M.cuda(), cube.cuda()

            results, spatial_weight, _ = self.net(img_rgb, img, pcl, self.evalData, center, M, cube, cam_para, 0.8)

            batch_size = img.size(0)
            mano_mesh = torch.zeros([img.size(0), 779, 3]).to(img.device)
            joint_xyz_list = []
            for index, stage_type in enumerate(self.config.stage_type):
                if stage_type == 0:
                    joint_uvd = results[index]
                    joint_xyz = self.evalData.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para)
                elif stage_type == 1:

                    pixel_pd = results[index]
                    joint_uvd = self.GFM_.feature2joint(img, pixel_pd, self.config.feature_type,
                                                        self.config.feature_para)
                    joint_xyz = self.evalData.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para)

                elif stage_type == 2 or stage_type == 3:
                    joint_xyz = results[index]
                    joint_uvd = self.evalData.xyz_nl2uvdnl_tensor(joint_xyz, center, M, cube, cam_para)

                joint_xyz_world = joint_xyz * cube.unsqueeze(1) / 2 + center.unsqueeze(1)

                joint_xyz_list.append(joint_xyz)

                if ii % 1 == 0:
                    img_show = vis_tool.draw_2d_pose(img[0], joint_uvd[0], self.dataset)
                    self.writer.add_image('eval_img_{}'.format(index), np.transpose(img_show, (2, 0, 1)) / 255.0,
                                          global_step=ii)
                    rgb_show = vis_tool.draw_2d_pose_rgb(img_rgb[0], joint_uvd[0], self.dataset)
                    rgb_show = cv2.cvtColor(rgb_show, cv2.COLOR_BGR2RGB)
                    self.writer.add_image('eval_img_rgb{}'.format(index), np.transpose(rgb_show, (2, 0, 1)) / 255.0,
                                          global_step=ii)

            joint_xyz_world = joint_xyz_list[-1] * cube.unsqueeze(1) / 2 + center.unsqueeze(1)
            mesh_xyz_world = mano_mesh * cube.unsqueeze(1) / 2 + center.unsqueeze(1)
            mesh_xyz_world = mesh_xyz_world.detach().cpu()
            joint_xyz_world = joint_xyz_world.detach().cpu().numpy()[:, MANO2HO3D, :]
            joint_xyz_world *= np.array([1, -1, -1]) / 1000
            mesh_xyz_world *= np.array([1, -1, -1]) / 1000
            joint_list = joint_list + np.split(joint_xyz_world, batch_size, axis=0)
            mesh_list = mesh_list + np.split(mesh_xyz_world, batch_size, axis=0)
        self.dump(self.model_dir + '/pred.json', joint_list, mesh_list)
        return 0

    @torch.no_grad()
    def xyz2error(self, output, joint, center, cube_size, write_file=None):
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
        if joint_num == 23:
            calculate = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 21, 22, 20]
            errors = np.sqrt(np.sum(errors[:, calculate, :], axis=2))
        else:
            errors = np.sqrt(np.sum(errors, axis=2))
        return errors

    @torch.no_grad()
    def z2error(self, output, joint, center, cube_size, write_file=None):
        output = output.detach().cpu().numpy()
        joint = joint.detach().cpu().numpy()
        center = center.detach().cpu().numpy()
        cube_size = cube_size.detach().cpu().numpy()
        batchsize, joint_num, _ = output.shape
        center = np.tile(center.reshape(batchsize, 1, -1), [1, joint_num, 1])
        cube_size = np.tile(cube_size.reshape(batchsize, 1, -1), [1, joint_num, 1])

        joint_xyz = output * cube_size / 2 + center
        joint_world_select = joint * cube_size / 2 + center

        # errors = (joint_xyz - joint_world_select) * (joint_xyz - joint_world_select)
        # errors = np.sqrt(np.sum(errors, axis=2))
        # print(joint_xyz.shape)
        depth_errors = abs((joint_xyz - joint_world_select)[:, :, 2])
        return depth_errors

    @torch.no_grad()
    def xy2error(self, output, joint, center, cube_size, write_file=None):
        output = output.detach().cpu().numpy()
        joint = joint.detach().cpu().numpy()
        center = center.detach().cpu().numpy()
        cube_size = cube_size.detach().cpu().numpy()
        batchsize, joint_num, _ = output.shape
        center = np.tile(center.reshape(batchsize, 1, -1), [1, joint_num, 1])
        cube_size = np.tile(cube_size.reshape(batchsize, 1, -1), [1, joint_num, 1])

        joint_xyz = output * cube_size / 2 + center
        joint_world_select = joint * cube_size / 2 + center

        xy_errors = (joint_xyz - joint_world_select)[:, :, :2] * (joint_xyz - joint_world_select)[:, :, :2]
        xy_errors = np.sqrt(np.sum(xy_errors, axis=2))
        # print(joint_xyz.shape)
        # depth_errors = abs((joint_xyz - joint_world_select)[:, :, 2])
        return xy_errors

    def dump(self, pred_out_path, xyz_pred_list, verts_pred_list):
        """ Save predictions into a json file. """
        # make sure its only lists
        xyz_pred_list = [x[0].tolist() for x in xyz_pred_list]
        verts_pred_list = [x[0].tolist() for x in verts_pred_list]

        # save to a json
        with open(pred_out_path, 'w') as fo:
            json.dump(
                [
                    xyz_pred_list,
                    verts_pred_list
                ], fo)
        print('Dumped %d joints and %d verts predictions to %s' % (
            len(xyz_pred_list), len(verts_pred_list), pred_out_path))


if __name__ == '__main__':
    # set_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    Trainer = Trainer(opt)
    if 'train' in Trainer.config.phase:
        Trainer.train()
        Trainer.test(epoch=-1)
        # Trainer.cal_FPS()
        Trainer.writer.close()
        Trainer.evalution()
    elif Trainer.config.phase == 'test':
        Trainer.test()
        # Trainer.result_file.close()
    elif Trainer.config.phase == 'eval':
        Trainer.evalution()
