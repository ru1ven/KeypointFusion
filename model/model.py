import math

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from model.resnet import BasicBlock, Bottleneck
import time
from model.transfusion_head import updatedDecoder
from util.generateFeature import GFM
from model.resnetUnet import OfficialResNetUnet, OfficialResNetUnet_RGB2offset_3D
from convNeXT.resnetUnet import convNeXTUnet, convNeXTUnet_RGB2offset_3D
import numpy as np
from pointnet2_ops import pointnet2_utils
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler, \
    BertConfig
from util.img2pcl import Pcl_utils

BN_MOMENTUM = 0.1

resnet = {18: (BasicBlock, [2, 2, 2, 2]),
          50: (Bottleneck, [3, 4, 6, 3]),
          101: (Bottleneck, [3, 4, 23, 3]),
          152: (Bottleneck, [3, 8, 36, 3])
          }


class TR_Encoder(BertPreTrainedModel):
    def __init__(self, config):
        super(TR_Encoder, self).__init__(config)

        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.img_dim = config.img_feature_dim
        self.img_embedding = nn.Linear(self.img_dim, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, img_feats, input_ids=None, token_type_ids=None, attention_mask=None,
                position_ids=None, head_mask=None):

        batch_size = len(img_feats)
        seq_length = len(img_feats[0])
        input_ids = torch.zeros([batch_size, seq_length], dtype=torch.long).cuda()

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        position_embeddings = self.position_embeddings(position_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError
        if self.config.multi:
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            if self.config.multi:
                head_mask = head_mask.to(
                    dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        img_embedding_output = self.img_embedding(img_feats)
        embeddings = position_embeddings + img_embedding_output
        embeddings = self.dropout(embeddings)
        encoder_outputs = self.encoder(embeddings,
                                       extended_attention_mask, head_mask=head_mask)
        sequence_output = encoder_outputs[0]

        outputs = (sequence_output,)
        if self.config.output_hidden_states:
            all_hidden_states = encoder_outputs[1]
            outputs = outputs + (all_hidden_states,)
        if self.config.output_attentions:
            all_attentions = encoder_outputs[-1]
            outputs = outputs + (all_attentions,)

        return outputs


class KP_Interaction_TR(BertPreTrainedModel):

    def __init__(self, config):
        super(KP_Interaction_TR, self).__init__(config)

        self.bert = TR_Encoder(config)
        self.cls_head = nn.Linear(config.hidden_size, config.output_feature_dim)
        self.residual = nn.Linear(config.img_feature_dim, config.output_feature_dim)
        self.init_weights()

    def forward(self, img_feats, input_ids=None, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, position_ids=None, head_mask=None):
        predictions = self.bert(img_feats=img_feats, input_ids=input_ids, position_ids=position_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask, head_mask=head_mask)

        pred_score = self.cls_head(predictions[0])
        res_img_feats = self.residual(img_feats)
        pred_score = pred_score + res_img_feats

        return predictions[0], pred_score


class DESA(nn.Module):
    def __init__(self, in_channel, mlp, S=[64, 64, 64], radius=[0.1, 0.2, 0, 4]):
        super(DESA, self).__init__()
        self.S = S
        self.radius = radius
        self.scale_num = len(radius)
        self.groupers = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        self.conv_l0_blocks = nn.ModuleList()
        self.conv_f0_blocks = nn.ModuleList()
        self.bn_l0_blocks = nn.ModuleList()
        self.bn_f0_blocks = nn.ModuleList()

        for i in range(self.scale_num):
            self.conv_l0_blocks.append(nn.Conv2d(3, mlp[0], 1))
            self.conv_f0_blocks.append(nn.Conv2d(in_channel, mlp[0], 1))
            self.bn_l0_blocks.append(nn.BatchNorm2d(mlp[0]))
            self.bn_f0_blocks.append(nn.BatchNorm2d(mlp[0]))
            last_channel = mlp[0]
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            for out_channel in mlp[1:]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius[i], S[i], use_xyz=True))

        self.fusion = nn.Sequential(
            nn.Conv1d(in_channel + mlp[-1] * self.scale_num, in_channel, 1),
            nn.BatchNorm1d(in_channel),
            nn.ReLU()
        )

    def forward(self, pcl_feat, node_feat, pcl_xyz, node_xyz):
        B, J, C = node_feat.size()
        pcl_xyz_expand = torch.cat((pcl_xyz, node_xyz), dim=1)
        pcl_feat_expand = torch.cat((pcl_feat, node_feat), dim=1)
        grouped_feat_list = []
        for i in range(self.scale_num):
            radius = self.radius[i]
            grouper = self.groupers[i]
            grouped_feat = grouper(pcl_xyz_expand, node_xyz, pcl_feat_expand.transpose(2, 1).contiguous())
            grouped_xyz, grouped_feat = torch.split(grouped_feat, [3, C], dim=1)

            group_xyz_norm = grouped_xyz / radius
            grouped_feat = grouped_feat - node_feat.permute(0, 2, 1).view(B, C, J, 1)
            # grouped_feat = torch.cat([grouped_xyz, grouped_feat], dim=1)

            # group_idx = query_ball_point(radius, S, pcl_xyz_new, node_xyz)
            # grouped_xyz = index_points(pcl_xyz_new, group_idx)
            # grouped_feat = index_points(pcl_feat_new, group_idx)
            # grouped_xyz = (grouped_xyz - node_xyz.view(B, J, 1, 3)) / radius
            # grouped_feat = grouped_feat - node_feat.view(B, J, 1, C)
            # grouped_feat = torch.cat((grouped_feat, grouped_xyz), dim=-1).permute(0, 3, 1, 2)

            # init layer
            loc = self.bn_l0_blocks[i](self.conv_l0_blocks[i](group_xyz_norm))
            feat = self.bn_f0_blocks[i](self.conv_f0_blocks[i](grouped_feat))
            grouped_feat = loc + feat
            grouped_feat = F.relu(grouped_feat)

            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_feat = F.relu(bn(conv(grouped_feat)))
            grouped_feat = torch.max(grouped_feat, dim=-1)[0]
            grouped_feat_list.append(grouped_feat)

        grouped_feat_list.append(node_feat.permute(0, 2, 1))
        grouped_feat_concat = torch.cat(grouped_feat_list, dim=1)
        out = self.fusion(grouped_feat_concat)
        return out.permute(0, 2, 1)


class Block_KPFusion(nn.Module):

    def __init__(self, joint_num=21, feature_size=128, num_points=4):
        super(Block_KPFusion, self).__init__()
        self.joint_num = joint_num

        self.dim = 128
        self.feature_size = feature_size
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(in_features=self.feature_size, out_features=2 * self.num_points, bias=True)
        self.attention_weights = nn.Linear(in_features=self.feature_size, out_features=self.num_points, bias=True)
        self.sampling_feature_embding = nn.Linear(in_features=self.dim, out_features=self.dim, bias=True)

        self.FA = DESA(128, [128, 128], [64, 64, 64], [0.1, 0.2, 0.4])
        config_init = BertConfig.from_pretrained("./config/")

        config_init.output_attentions = False
        config_init.hidden_dropout_prob = 0.1
        config_init.img_feature_dim = 128
        config_init.output_feature_dim = 3
        config_init.num_hidden_layers = 4
        config_init.hidden_size = 128
        config_init.num_attention_heads = 4
        config_init.intermediate_size = config_init.num_attention_heads * 4
        config_init.multi = False
        self.init_TR = KP_Interaction_TR(config_init)

        config_final = BertConfig.from_pretrained("./config/")
        config_final.output_attentions = False
        config_final.hidden_dropout_prob = 0.1
        config_final.img_feature_dim = 131
        config_final.output_feature_dim = 3
        config_final.num_hidden_layers = 4
        config_final.hidden_size = 128
        config_final.num_attention_heads = 4
        config_final.intermediate_size = config_final.num_attention_heads * 4
        config_final.multi = False
        self.final_TR = KP_Interaction_TR(config_final)
        self.crossTR = updatedDecoder(joint_num=joint_num,
                                      hidden_channel=config_final.hidden_size,
                                      num_heads=config_final.num_attention_heads,
                                      ffn_channel=config_final.hidden_size,
                                      dropout=config_final.hidden_dropout_prob,
                                      num_decoder_layers=config_final.num_hidden_layers,
                                      activation='relu')

        self.pcl_feat_emb = nn.Sequential(nn.Conv1d(self.dim, self.dim, 1), nn.BatchNorm1d(self.dim))
        self.pcl_xyz_emb = nn.Sequential(nn.Conv1d(3, self.dim, 1), nn.BatchNorm1d(self.dim))
        self.pcl_pose_emb = nn.Sequential(nn.Conv1d(self.joint_num * 5, self.dim, 1), nn.BatchNorm1d(self.dim))
        self.joint_feat_emb = nn.Sequential(nn.Conv1d(self.dim, self.dim, 1), nn.BatchNorm1d(self.dim))
        self.joint_xyz_emb = nn.Sequential(nn.Conv1d(3, self.dim, 1), nn.BatchNorm1d(self.dim))
        self.pcl_feat_emb_RGB = nn.Sequential(nn.Conv1d(self.dim, self.dim, 1), nn.BatchNorm1d(self.dim))

        self.sigmoid = nn.Sigmoid()
        self.atten_spatial = nn.Conv2d(in_channels=feature_size + joint_num, out_channels=joint_num, kernel_size=1,
                                       stride=1, bias=True)
        self.fc_spatial2joint_feature = nn.Linear(in_features=32 * 32, out_features=1, bias=True)
        self.reduction_joint_feature = nn.Linear(in_features=self.dim * 2, out_features=self.dim, bias=True)
        self.reduction_joint_feature_update = nn.Conv1d(in_channels=joint_num * 3, out_channels=joint_num,
                                                        kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=-1)
        self.apply(self._init_weights)
        self.cls_head = nn.Linear(config_final.hidden_size, config_final.output_feature_dim)
        self.weight_dis = nn.Parameter(torch.zeros([1]))

        self.GFM_ = GFM()

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.001)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=0.001)

    def forward(self, img_feat, img_feature_rgb, pcl, joint_xyz, pcl_closeness, pcl_index, img_offset,
                updated_2d_feature, loader, img_down, center, M, cube, cam_para, writer=None, ii=0):

        B, N, _ = pcl.size()
        B, C, H, W = img_feat.size()
        J = self.joint_num

        # RGB Keypoint Aggregation
        pcl_offset_xyz = pcl_joint2offset(joint_xyz, pcl, 0.8)

        pcl_feat_index = pcl_index.view(B, 1, -1).repeat(1, C, 1)  # B*128*(K*1024)
        pcl_feat = torch.gather(img_feat.view(B, C, -1), -1, pcl_feat_index).view(B, C, N, -1)
        pcl_feat = torch.sum(pcl_feat * pcl_closeness.unsqueeze(1), dim=-1).permute(0, 2, 1)
        pcl_feat_RGB = torch.gather(img_feature_rgb.view(B, C, -1), -1, pcl_feat_index).view(B, C, N, -1)
        pcl_feat_RGB = torch.sum(pcl_feat_RGB * pcl_closeness.unsqueeze(1), dim=-1).permute(0, 2, 1)

        """index token coordinate"""
        pcl_index_weight = pcl_index.view(B, 1, -1).repeat(1, J, 1)
        pcl_weight = torch.gather(img_offset[:, J * 4:, :, :].view(B, J, -1), -1, pcl_index_weight).view(B, J, N, -1)
        pcl_weight = torch.sum(pcl_weight * pcl_closeness.unsqueeze(1), dim=-1).permute(0, 2, 1)

        pcl_weight = pcl_weight.detach()  # B S N
        pcl_offset_xyz = pcl_offset_xyz.detach()  # B S N*4(offset+dis)

        # Decoupled Generation of RGB-D Point Cloud Features
        pcl_feat = self.pcl_feat_emb(pcl_feat.permute(0, 2, 1)).permute(0, 2, 1) + \
                   self.pcl_xyz_emb(pcl.permute(0, 2, 1)).permute(0, 2, 1) + \
                   self.pcl_pose_emb(torch.cat((pcl_weight, pcl_offset_xyz), dim=-1).permute(0, 2, 1)).permute(0, 2, 1)
        pcl_feat = F.relu(pcl_feat)

        pcl_feat = F.relu(pcl_feat + self.pcl_feat_emb_RGB(pcl_feat_RGB.permute(0, 2, 1)).permute(0, 2, 1))

        attention = F.softmax(pcl_weight.permute(0, 2, 1), dim=-1)
        joint_feat = torch.matmul(attention, pcl_feat)

        # Aggregate joint_feat based on refined joint_xyz
        joint_feat = self.joint_feat_emb(joint_feat.permute(0, 2, 1)).permute(0, 2, 1) + \
                     self.joint_xyz_emb(joint_xyz.detach().permute(0, 2, 1)).permute(0, 2, 1)
        joint_feat = F.relu(joint_feat)
        # B*J*C
        joint_feat = self.FA(pcl_feat, joint_feat, pcl, joint_xyz.detach())

        # Intra-modal Keypoint Feature Interaction
        outfeature_init_TR, refined_3d_joints = self.init_TR(joint_feat)

        # Depth Keypoint Aggregation
        # Compute Heatmap & Geometry Adjacency Map
        hm = self.GFM_.joint2heatmap(refined_3d_joints[:, :, :2], 0.8, H, sigma=1)
        GAM = loader.img2anchor_dis(refined_3d_joints, img_down, center, M, cube, cam_para)  # B*J*H*W
        spatial_weight_loss = self.sigmoid(self.atten_spatial(torch.cat([img_feature_rgb, hm], dim=1)))
        img_feature_rgb = F.relu((F.sigmoid(self.weight_dis) * GAM.unsqueeze(2) + (
                1 - F.sigmoid(self.weight_dis)) * spatial_weight_loss.unsqueeze(2)) * img_feature_rgb.unsqueeze(1))

        img_feature_rgb = img_feature_rgb.view(B, J, C, -1)
        img_feat_j = self.fc_spatial2joint_feature(img_feature_rgb)
        img_feat_j = img_feat_j.view(B, J, C)
        if updated_2d_feature is not None:
            img_feat_j = F.relu((img_feat_j + updated_2d_feature) / 2)

        # Inter-modal Keypoint Feature Interaction
        refined_joint_feat = self.crossTR(img_feat_j, outfeature_init_TR).permute(0, 2, 1)
        refined_joint_feat = torch.cat([refined_3d_joints, refined_joint_feat], dim=2)
        outfeature_final_TR, refined_2d_joints = self.final_TR(refined_joint_feat)

        return refined_3d_joints, refined_2d_joints, img_feat_j, spatial_weight_loss, None


class KPFusion(nn.Module):
    def __init__(self, net, pretrain, joint_num, dataset, mano_dir, kernel_size=1):
        super(KPFusion, self).__init__()
        self.joint_num = joint_num
        self.kernel_size = kernel_size
        self.dim = 128
        self.classify_out = 3
        self.num_stages = 2
        self.net = net
        if 'convnext' in self.net:
            self.backbone_rgb = convNeXTUnet_RGB2offset_3D(net, joint_num, pretrain=pretrain, deconv_dim=self.dim,
                                                           out_dim_list=[joint_num * 3, joint_num, joint_num])

            self.backbone_d = convNeXTUnet(net, joint_num, pretrain=pretrain, deconv_dim=self.dim,
                                           out_dim_list=[joint_num * 3, joint_num, joint_num])
        else:
            self.backbone_rgb = OfficialResNetUnet_RGB2offset_3D(net, joint_num, pretrain=True, deconv_dim=self.dim,
                                                                 out_dim_list=[joint_num * 3, joint_num, joint_num])
            self.backbone_d = OfficialResNetUnet(net, joint_num, pretrain=True, deconv_dim=self.dim,
                                                 out_dim_list=[joint_num * 3, joint_num, joint_num])

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.pcl_utils = Pcl_utils()

        for i in range(self.num_stages):
            block = Block_KPFusion(joint_num=joint_num)
            setattr(self, f"block{i + 1}", block)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.001)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=0.001)

    def forward(self, img_rgb, img, pcl, loader, center, M, cube, cam_para, kernel=0.8, writer=None, ii=0):

        img_offset, img_feat = self.backbone_d(img)  # img_offset: B×C×W×H , C=2(direct vector)+1(heatmap)+1(weight)
        img_offset_rgb, img_feat_rgb = self.backbone_rgb(img_rgb)
        joint_uvd = offset2joint_weight(img_offset, img, kernel)
        B, C, H, W = img_feat.size()

        # initial result
        result = [img_offset, img_offset_rgb]
        joint_uvd = joint_uvd.detach()
        img_offset = img_offset.detach()

        # pcl feature
        B, N, _ = pcl.size()
        img_down = F.interpolate(img, [H, W])
        joint_xyz = loader.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para)
        pcl_closeness, pcl_index = loader.img2pcl_index(pcl, img_down, center, M, cube, cam_para, select_num=4)

        updated_2d_feature = [None] * (self.num_stages + 1)
        spatial_weight = [None] * self.num_stages

        # 2-stage RGB-D KFAM
        for i in range(self.num_stages):
            block = getattr(self, f"block{i + 1}")
            refined_3d_joints, refined_2d_joints, updated_2d_feature[i + 1], \
            spatial_weight[i], _ = block(img_feat, img_feat_rgb, pcl, joint_xyz, pcl_closeness, pcl_index, img_offset,
                                         updated_2d_feature[i], loader, img_down, center, M, cube, cam_para, writer, ii)
            result.append(refined_3d_joints)
            result.append(refined_2d_joints)
            joint_xyz = refined_2d_joints

        return result, spatial_weight, None


def img2pcl(img):
    B, _, W, H = img.size()
    device = img.device
    mesh_x = 2.0 * (torch.arange(W).unsqueeze(1).expand(W, W).float() + 0.5) / W - 1.0
    mesh_y = 2.0 * (torch.arange(W).unsqueeze(0).expand(W, W).float() + 0.5) / W - 1.0
    coords = torch.stack((mesh_y, mesh_x), dim=0)
    coords = torch.unsqueeze(coords, dim=0).repeat(B, 1, 1, 1).to(device)
    img_uvd = torch.cat((coords, img), dim=1).view(B, 3, H * W).permute(0, 2, 1)
    return img_uvd


def joint2offset(joint, img, kernel_size, feature_size):
    device = joint.device
    batch_size, _, img_height, img_width = img.size()
    img = F.interpolate(img, size=[feature_size, feature_size])
    _, joint_num, _ = joint.view(batch_size, -1, 3).size()
    joint_feature = joint.reshape(joint.size(0), -1, 1, 1).repeat(1, 1, feature_size, feature_size)
    mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size,
                                                                   feature_size).float() + 0.5) / feature_size - 1.0
    mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size,
                                                                   feature_size).float() + 0.5) / feature_size - 1.0
    coords = torch.stack((mesh_y, mesh_x), dim=0)
    coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
    coords = torch.cat((coords, img), dim=1).repeat(1, joint_num, 1, 1)
    offset = joint_feature - coords
    offset = offset.view(batch_size, joint_num, 3, feature_size, feature_size)
    dist = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=2))
    offset_norm = (offset / (dist.unsqueeze(2)))

    heatmap = (kernel_size - dist) / kernel_size

    mask = heatmap.ge(0).float() * img.lt(0.99).float().view(batch_size, 1, feature_size, feature_size)
    offset_norm_mask = (offset_norm * mask.unsqueeze(2)).view(batch_size, -1, feature_size, feature_size)
    heatmap_mask = heatmap * mask.float()
    return torch.cat((offset_norm_mask, heatmap_mask), dim=1)


def offset2joint_weight(offset, depth, kernel_size):
    device = offset.device
    batch_size, joint_num, feature_size, feature_size = offset.size()
    joint_num = int(joint_num / 5)
    if depth.size(-1) != feature_size:  # 下采样深度图
        depth = F.interpolate(depth, size=[feature_size, feature_size])

    offset_unit = offset[:, :joint_num * 3, :, :].contiguous()  # b * (3*J) * fs * fs
    heatmap = offset[:, joint_num * 3:joint_num * 4, :, :].contiguous()
    weight = offset[:, joint_num * 4:, :, :].contiguous()

    mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size,
                                                                   feature_size).float() + 0.5) / feature_size - 1.0
    mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size,
                                                                   feature_size).float() + 0.5) / feature_size - 1.0
    coords = torch.stack((mesh_y, mesh_x), dim=0)
    coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
    coords = torch.cat((coords, depth), dim=1).repeat(1, joint_num, 1, 1).view(batch_size, joint_num, 3, -1)

    mask = depth.lt(0.99).float()
    offset_mask = (offset_unit * mask).view(batch_size, joint_num, 3, -1)  # 截取深度图中有值的部分
    heatmap_mask = (heatmap * mask).view(batch_size, joint_num, -1)
    weight_mask = weight.masked_fill(depth.gt(0.99), -1e8)
    normal_weight = F.softmax(weight_mask.view(batch_size, joint_num, -1), dim=-1)  # b * J * fs^2

    if torch.is_tensor(kernel_size):
        kernel_size = kernel_size.to(device)
        dist = kernel_size.view(1, joint_num, 1) - heatmap_mask * kernel_size.view(1, joint_num, 1)
    else:
        dist = kernel_size - heatmap_mask * kernel_size

    joint = torch.sum(
        (offset_mask * dist.unsqueeze(2).repeat(1, 1, 3, 1) + coords) * normal_weight.unsqueeze(2).repeat(1, 1, 3, 1),
        dim=-1)
    return joint


def pcl_joint2offset(joint, pcl, kernel_size):
    """
    :param: joint BxJx3--xyz坐标
    :param: pcl BxNx3
    """
    B, J, _ = joint.size()
    N = pcl.size(1)
    device = joint.device
    offset = joint.unsqueeze(2) - pcl.unsqueeze(1)  # B J 1 3 - B 1 N 3 -> B J N 3
    dis = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=-1))  # B J N
    offset_norm = offset / (dis.unsqueeze(-1) + 1e-8)
    offset_norm = offset_norm.permute(0, 1, 3, 2).reshape(B, J * 3, N)

    if torch.is_tensor(kernel_size):
        kernel_size = kernel_size.to(device)
        dis = (kernel_size.view(1, J, 1) - dis) / kernel_size.view(1, J, 1)
    else:
        dis = (kernel_size - dis) / kernel_size

    mask = dis.ge(0).float() * pcl[:, :, 2].lt(0.99).float().unsqueeze(1)
    dis = dis * mask  # closeness map
    offset_norm = offset_norm * mask.view(B, J, 1, N).repeat(1, 1, 3, 1).reshape(B, -1, N)  # 3D directional unit vector
    return torch.cat((offset_norm, dis), dim=1).to(device).permute(0, 2, 1)


def pcl_offset2joint_weight(pcl_result, pcl, kernel_size):
    """
    :param: pcl_result BxNx(5*J)
    :param: pcl BxNx3
    """
    assert pcl.size(2) == 3
    pcl_result = pcl_result.permute(0, 2, 1)
    B, J, N = pcl_result.size()
    J = int(J / 5)
    device = pcl.device

    coords = pcl.permute(0, 2, 1).reshape(B, 1, 3, N)
    offset = pcl_result[:, :J * 3, :].view(B, J, 3, N)
    heatmap = pcl_result[:, J * 3:J * 4, :].view(B, J, 1, N)
    weight = pcl_result[:, J * 4:, :].view(B, J, 1, N)

    mask = pcl[:, :, 2].gt(0.99).view(B, 1, 1, N)
    weight_mask = torch.masked_fill(weight, mask, -1e8)
    normal_weight = F.softmax(weight_mask, dim=-1)

    if torch.is_tensor(kernel_size):
        kernel_size = kernel_size.to(device)
        dist = kernel_size.view(1, J, 1) - heatmap * kernel_size.view(1, J, 1)
    else:
        dist = kernel_size - heatmap * kernel_size

    joint = torch.sum((offset * dist + coords) * normal_weight, dim=-1)
    return joint


def offset2joint_weight_2D(offset, kernel_size=None, hand_mask=None, joint_nb=21, beta=15):
    device = offset.device
    B, J, S, S = offset.size()
    J = joint_nb

    offset_unit = offset[:, :J * 2, :, :].view(B, J, 2, -1).contiguous()
    dis = offset[:, J * 2:J * 3, :, :].view(B, J, -1).contiguous()
    weight = offset[:, J * 3:J * 4, :, :].view(B, J, -1).contiguous()

    mesh_x = (torch.arange(S).unsqueeze(1).expand(S, S).float() + 0.5) / S
    mesh_y = (torch.arange(S).unsqueeze(0).expand(S, S).float() + 0.5) / S
    coords = torch.stack((mesh_y, mesh_x), dim=0)
    coords = torch.unsqueeze(coords, dim=0).repeat(B, J, 1, 1).view(B, J, 2, -1).to(device)
    # weight = self.softmax(weight * beta)
    weight = F.softmax(weight * beta, dim=-1)  # b * J * fs^2
    # mask = dis.gt(0).unsqueeze(2)
    dis = kernel_size - dis * kernel_size
    if hand_mask is not None:
        hand_mask = hand_mask.reshape(B, J, 1, -1)  # B J 1 S S
        joint = torch.sum((offset_unit * dis.unsqueeze(2).repeat(1, 1, 2, 1) * hand_mask + coords * hand_mask) *
                          weight.unsqueeze(2).repeat(1, 1, 2, 1), dim=-1)
    else:
        joint = torch.sum(
            (offset_unit * dis.unsqueeze(2).repeat(1, 1, 2, 1) + coords) * weight.unsqueeze(2).repeat(1, 1, 2, 1),
            dim=-1)
    # joint = torch.sum((offset_unit * dis.unsqueeze(2).repeat(1, 1, 2, 1) * mask + coords * mask), dim=-1) /\
    #                   torch.sum(mask, dim=-1)
    return joint


# def getpcl(imgD, com3D, cube, M, cam_para=None):
#     mask = torch.isclose(imgD, torch.tensor(1.))
#     print(imgD.shape)
#     print(cube.shape)
#     print(com3D.shape)
#     dpt_ori = imgD * cube[:,2].unsqueeze(1).unsqueeze(1) / 2.0 + com3D[:,2].unsqueeze(1).unsqueeze(1)
#     # change the background value
#     dpt_ori[mask] = 0
#
#     pcl = (depthToPCL(dpt_ori.detach().cpu().numpy(), M.detach().cpu().numpy(), cam_para) - com3D.detach().cpu().numpy())
#     pcl_num = pcl.shape[0]
#     cube_tile = (cube / 2.0).repeat(1,pcl_num).reshape([cube.shape[0],pcl_num, 3])
#     pcl = pcl / cube_tile
#     return pcl

def depthToPCL(dpt, T, paras=None, background_val=0.):
    fx, fy, fu, fv = paras

    # get valid points and transform
    pts = np.asarray(np.where(~np.isclose(dpt, background_val))).transpose()
    pts = np.concatenate([pts[:, [1, 0]] + 0.5, np.ones((pts.shape[0], 1), dtype='float32')], axis=1)
    pts = np.dot(np.linalg.inv(np.asarray(T)), pts.T).T
    pts = (pts[:, 0:2] / pts[:, 2][:, None]).reshape((pts.shape[0], 2))

    # replace the invalid data
    depth = dpt[(~np.isclose(dpt, background_val))]

    # get x and y data in a vectorized way
    row = (pts[:, 0] - fu) / fx * depth
    col = 1 * (pts[:, 1] - fv) / fy * depth

    # combine x,y,depth
    return np.column_stack((row, col, depth))
