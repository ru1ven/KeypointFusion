import cv2
import numpy as np
import torch
import torch.nn as nn

class FilterLayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FilterLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y

'''
Feature Separation Part
'''

class FSP(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FSP, self).__init__()
        self.filter = FilterLayer(2 * in_planes, out_planes, reduction)

    def forward(self, guidePath, mainPath):
        combined = torch.cat((guidePath, mainPath), dim=1)
        channel_weight = self.filter(combined)
        out = mainPath + channel_weight * guidePath
        return out


class RGBDFusion(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16, bn_momentum=0.0003):
        self.init__ = super(RGBDFusion, self).__init__()
        self.in_planes = in_planes
        self.bn_momentum = bn_momentum

        self.fsp_rgb = FSP(in_planes, out_planes, reduction)
        self.fsp_depth = FSP(in_planes, out_planes, reduction)

        self.gate_rgb = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)
        self.gate_depth = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x,train_writer=None,global_step=0,layer_stage=0):
        rgb, depth = x

        cat_fea = torch.cat([rgb, depth], dim=1)

        attention_vector_l = self.gate_rgb(cat_fea)
        attention_vector_r = self.gate_depth(cat_fea)

        attention_vector = torch.cat([attention_vector_l, attention_vector_r], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector_l, attention_vector_r = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]

        if train_writer is not None:
            #train_writer.add_image('attention_RGB_stage{}'.format(layer_stage), attention_vector_l[0],global_step=global_step)
            #train_writer.add_image('attention_Depth_stage{}'.format(layer_stage), attention_vector_r[0],global_step=global_step)
            train_writer.add_scalar('RGB_weight_fusion_stage{}'.format(layer_stage), attention_vector_l.mean(0).mean(0).mean(0).mean(0).detach(),global_step)
            train_writer.add_scalar('Depth_weight_fusion_stage{}'.format(layer_stage),attention_vector_r.mean(0).mean(0).mean(0).mean(0).detach(), global_step)

        merge_feature = rgb * attention_vector_l + depth * attention_vector_r

        rgb_out = (rgb + merge_feature) / 2
        depth_out = (depth + merge_feature) / 2
        #depth_out = depth

        rgb_out = self.relu1(rgb_out)
        depth_out = self.relu2(depth_out)

        return [rgb_out, depth_out], merge_feature



class ACFusion(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16, bn_momentum=0.0003):
        self.init__ = super(ACFusion, self).__init__()
        self.in_planes = in_planes
        self.bn_momentum = bn_momentum

        self.in_planes = in_planes
        self.cam_rgb = nn.Conv2d(in_planes, in_planes, kernel_size=1, bias=True)
        self.cam_depth = nn.Conv2d(in_planes, in_planes, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x,train_writer=None,global_step=0,layer_stage=0):
        rgb, depth = x
        w_rgb = self.cam_rgb(self.pool(rgb))
        w_d = self.cam_depth(self.pool(depth))
        w_rgb=self.sigmoid(w_rgb)
        w_d = self.sigmoid(w_d)
        rec_rgb = w_rgb*rgb
        rec_d = w_d * depth

        merge_feature = rec_rgb+rec_d
        rgb_out = (rgb + merge_feature) / 2
        depth_out = (depth + merge_feature) / 2
        rgb_out = self.relu1(rgb_out)
        depth_out = self.relu2(depth_out)

        return [rgb_out, depth_out], merge_feature

