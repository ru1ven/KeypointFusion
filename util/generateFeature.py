from math import sqrt

import numpy as np
import torch
from scipy.ndimage.morphology import distance_transform_edt
import torch.nn.functional as F

class GFM:
    def __init__(self):
        self.softmax = torch.nn.Softmax(dim=-1)

    def joint2heatmap2d(self, joint, img, std, heatmap_size):
        # joint depth is norm[-1,1]
        divce = joint.device
        img_down = F.interpolate(img, size=[heatmap_size, heatmap_size])
        batch_size, joint_num, _ = joint.size()
        xx, yy = np.meshgrid(np.arange(heatmap_size), np.arange(heatmap_size))
        mesh_x = torch.from_numpy(xx).view(1, 1, heatmap_size, heatmap_size).repeat(batch_size, joint_num, 1, 1).float().to(divce)+0.5
        mesh_y = torch.from_numpy(yy).view(1, 1, heatmap_size, heatmap_size).repeat(batch_size, joint_num, 1, 1).float().to(divce)+0.5
        joint_ht = torch.zeros_like(joint).to(divce)
        joint_ht[:, :, 0] = (joint[:, :, 0] + 1) / 2 * heatmap_size
        joint_ht[:, :, 1] = (joint[:, :, 1] + 1) / 2 * heatmap_size
        joint_x = joint_ht[:, :, 0].view(batch_size, joint_num, 1, 1).repeat(1, 1, heatmap_size, heatmap_size).float()
        joint_y = joint_ht[:, :, 1].view(batch_size, joint_num, 1, 1).repeat(1, 1, heatmap_size, heatmap_size).float()
        heatmap = torch.exp(-(torch.pow((mesh_x - joint_x) / std, 2) + torch.pow((mesh_y.to(divce) - joint_y) / std, 2)))
        mask = heatmap.ge(0.01).float() * img_down.lt(0.99).float().view(batch_size, 1, heatmap_size, heatmap_size)
        return heatmap, mask

    def joint2plainoffset(self, joint, img, kernel_size, feature_size):
        device = joint.device
        batch_size, _, img_height, img_width = img.size()
        img = F.interpolate(img,size=[feature_size,feature_size])
        batch_size,joint_num,_ = joint.view(batch_size,-1,3).size()
        joint_feature = joint[:,:,:2].contiguous().view(batch_size,joint_num*2,1,1).repeat(1,1,feature_size,feature_size)
        mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        coords = torch.stack((mesh_y,mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).repeat(1, joint_num, 1, 1).to(device)
        offset = joint_feature - coords
        offset = offset.view(batch_size,joint_num,2,feature_size,feature_size)
        dist = torch.sqrt(torch.sum(torch.pow(offset,2),dim=2)+1e-8)
        offset_norm = (offset / (dist.unsqueeze(2)))
        heatmap = (kernel_size - dist)/kernel_size
        mask = heatmap.ge(0).float() * img.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        offset_norm_mask = (offset_norm*mask.unsqueeze(2)).view(batch_size,-1,feature_size,feature_size)
        heatmap_mask = heatmap * mask.float()
        return torch.cat((offset_norm_mask, heatmap_mask),dim=1), mask

    def joint2depthoffset(self, joint, img, mask, feature_size):
        batch_size, _, img_height, img_width = img.size()
        img = F.interpolate(img,size=[feature_size,feature_size])
        batch_size,joint_num,_ = joint.view(batch_size,-1,3).size()
        joint_feature = joint[:,:,2].contiguous().view(batch_size,joint_num,1,1).repeat(1,1,feature_size,feature_size)
        offset = joint_feature - img.view(batch_size,1,feature_size,feature_size)
        offset = offset.view(batch_size,joint_num,1,feature_size,feature_size)
        offset_mask = (offset*mask.unsqueeze(2)).view(batch_size,-1,feature_size,feature_size)
        return offset_mask

    def joint2offset(self, joint, img, kernel_size, feature_size):
        device = joint.device
        batch_size, _, img_height, img_width = img.size()
        img = F.interpolate(img,size=[feature_size,feature_size])
        _,joint_num,_ = joint.view(batch_size,-1,3).size()
        joint_feature = joint.reshape(joint.size(0),-1,1,1).repeat(1,1,feature_size,feature_size)
        # mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        # mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords, img), dim=1).repeat(1, joint_num, 1, 1)
        offset = joint_feature - coords
        offset = offset.view(batch_size, joint_num, 3, feature_size, feature_size)
        dist = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=2) + 1e-8)
        offset_norm = (offset / (dist.unsqueeze(2)))
        if torch.is_tensor(kernel_size):
            kernel_size = kernel_size.to(device)
            heatmap = (kernel_size.view(1, joint_num, 1, 1) - dist) / kernel_size.view(1, joint_num, 1, 1)
        else:
            heatmap = (kernel_size - dist)/kernel_size
        mask = heatmap.ge(0).float() * img.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        offset_norm_mask = (offset_norm * mask.unsqueeze(2)).view(batch_size, -1, feature_size, feature_size).float()
        heatmap_mask = heatmap * mask.float()
        return torch.cat((offset_norm_mask, heatmap_mask),dim=1)
        # return torch.cat((offset_norm.view(batch_size,-1,feature_size,feature_size), heatmap),dim=1).float()

    def offset2joint(self, offset, depth, kernel_size, topk=30):
        device = offset.device
        batch_size, joint_num, feature_size, feature_size = offset.size()
        joint_num = int(joint_num / 4)
        if depth.size(-1) != feature_size:
            depth = F.interpolate(depth, size=[feature_size, feature_size])
        offset_unit = offset[:, :joint_num*3, :, :].contiguous().view(batch_size,joint_num,3,-1)
        heatmap = offset[:, joint_num*3:, :, :].contiguous().view(batch_size,joint_num,-1)
        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords, depth), dim=1).repeat(1, joint_num, 1, 1).view(batch_size, joint_num, 3, -1)
        value, index = torch.topk(heatmap, topk, dim=-1)
        index = index.unsqueeze(2).repeat(1,1,3,1)
        value = value.unsqueeze(2).repeat(1,1,3,1)
        offset_unit_select = torch.gather(offset_unit, -1, index)
        coords_select = torch.gather(coords, -1, index)
        if torch.is_tensor(kernel_size):
            kernel_size = kernel_size.to(device)
            dist = kernel_size.view(1, joint_num, 1, 1) - value * kernel_size.view(1, joint_num, 1, 1)
        else:
            dist = kernel_size - value * kernel_size

        joint = torch.sum((offset_unit_select*dist + coords_select)*value, dim=-1)
        joint = joint / (torch.sum(value, -1)+1e-8) # avoid 0
        return joint

    # change to with mask 2019/9/3
    def offset2joint_softmax(self, offset, depth, kernel_size, scale=30):
        device = offset.device
        batch_size, joint_num, feature_size, feature_size = offset.size()
        joint_num = int(joint_num / 4)
        if depth.size(-1) != feature_size:
            depth = F.interpolate(depth, size=[feature_size, feature_size])
        offset_unit = offset[:, :joint_num*3, :, :].contiguous()
        heatmap = offset[:, joint_num*3:, :, :].contiguous()
        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords, depth), dim=1).repeat(1, joint_num, 1, 1).view(batch_size, joint_num, 3, -1)
        mask = depth.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        offset_mask = (offset_unit * mask).view(batch_size,joint_num,3,-1)
        heatmap_mask = (heatmap * mask).view(batch_size, joint_num, -1)
        normal_heatmap = F.softmax(heatmap_mask * scale, dim=-1)

        dist = kernel_size - heatmap_mask * kernel_size
        joint = torch.sum((offset_mask * dist.unsqueeze(2).repeat(1,1,3,1) + coords) * normal_heatmap.unsqueeze(2).repeat(1,1,3,1), dim=-1)
        return joint

    # change to with mask 2019/9/3
    def offset2joint_selectsoftmax(self, offset, depth, kernel_size, scale=30, sample_num=1024):
        device = offset.device
        batch_size, joint_num, feature_size, feature_size = offset.size()
        joint_num = int(joint_num / 4)
        if depth.size(-1) != feature_size:
            depth = F.interpolate(depth, size=[feature_size, feature_size])
        offset_unit = offset[:, :joint_num*3, :, :].contiguous()
        heatmap = offset[:, joint_num*3:, :, :].contiguous()
        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords, depth), dim=1).repeat(1, joint_num, 1, 1).view(batch_size, joint_num, 3, -1)
        mask = depth.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        select_id = torch.multinomial(mask.view(batch_size,-1), sample_num, replacement=True).view(batch_size, 1, sample_num)
        offset_mask = (offset_unit*mask).view(batch_size,joint_num,3,-1)
        heatmap_mask = (heatmap*mask).view(batch_size, joint_num, -1)
        offset_select = torch.gather(offset_mask, -1, select_id.unsqueeze(1).repeat(1,joint_num,3,1))
        heatmap_select = torch.gather(heatmap_mask, -1, select_id.repeat(1,joint_num,1))
        coords_select = torch.gather(coords, -1, select_id.unsqueeze(1).repeat(1,joint_num,3,1))
        normal_heatmap = F.softmax(heatmap_select*scale, dim=-1)

        dist = kernel_size - heatmap_select * kernel_size
        joint = torch.sum((offset_select * dist.unsqueeze(2).repeat(1,1,3,1) + coords_select) * normal_heatmap.unsqueeze(2).repeat(1,1,3,1), dim=-1)
        return joint

    # change to with weight 2020/4/14
    def offset2joint_weight(self, offset, depth, kernel_size):
        device = offset.device
        batch_size, joint_num, feature_size, feature_size = offset.size()
        joint_num = int(joint_num / 5)
        if depth.size(-1) != feature_size:
            depth = F.interpolate(depth, size=[feature_size, feature_size])
        offset_unit = offset[:, :joint_num*3, :, :].contiguous()
        heatmap = offset[:, joint_num * 3:joint_num * 4, :, :].contiguous()
        weight = offset[:, joint_num * 4:, :, :].contiguous()
        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords, depth), dim=1).repeat(1, joint_num, 1, 1).view(batch_size, joint_num, 3, -1)

        mask = depth.lt(0.99).float()
        offset_mask = (offset_unit * mask).view(batch_size, joint_num, 3, -1)
        heatmap_mask = (heatmap * mask).view(batch_size, joint_num, -1)
        weight_mask = weight.masked_fill(depth.gt(0.99), -1e8)
        # weight_mask = torch.where(depth.lt(0.99), weight, -1e8 * torch.ones_like(weight).to(device))
        normal_weight = F.softmax(weight_mask.view(batch_size, joint_num, -1), dim=-1)

        if torch.is_tensor(kernel_size):
            kernel_size = kernel_size.to(device)
            dist = kernel_size.view(1, joint_num, 1) - heatmap_mask * kernel_size.view(1, joint_num, 1)
        else:
            dist = kernel_size - heatmap_mask * kernel_size

        joint = torch.sum((offset_mask * dist.unsqueeze(2).repeat(1,1,3,1) + coords) * normal_weight.unsqueeze(2).repeat(1,1,3,1), dim=-1)
        return joint

    # change to with weight 2020/4/14
    def offset2joint_weight_sample(self, offset, depth, kernel_size, scale=10):
        device = offset.device
        batch_size, joint_num, feature_size, feature_size = offset.size()
        joint_num = int(joint_num / 5)
        if depth.size(-1) != feature_size:
            depth = F.interpolate(depth, size=[feature_size, feature_size])
        offset_unit = offset[:, :joint_num*3, :, :].contiguous()
        heatmap = offset[:, joint_num * 3:joint_num * 4, :, :].contiguous()
        weight = offset[:, joint_num * 4:, :, :].contiguous()
        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords, depth), dim=1).repeat(1, joint_num, 1, 1).view(batch_size, joint_num, 3, -1)
        mask = depth.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        offset_mask = (offset_unit*mask).view(batch_size,joint_num,3,-1)
        heatmap_mask = (heatmap*mask).view(batch_size, joint_num, -1)
        weight_mask = (weight*mask).view(batch_size, joint_num, -1)
        normal_weigth = F.softmax(weight_mask*scale, dim=-1)

        sample_weight = torch.ones([batch_size, 64*64]).to(device)
        sample_weight = torch.where(depth.lt(0.99).view(batch_size, -1), sample_weight, torch.zeros_like(sample_weight).to(device))
        sample_index = torch.multinomial(sample_weight.view(batch_size, -1), 1024,replacement=True).unsqueeze(1)

        heatmap_mask = torch.gather(heatmap_mask, dim=-1, index=sample_index.repeat(1,21,1))
        offset_mask = torch.gather(offset_mask, dim=-1, index=sample_index.unsqueeze(1).repeat(1, 21, 3, 1))
        normal_weigth = torch.gather(normal_weigth, dim=-1, index=sample_index.repeat(1, 21, 1))
        coords = torch.gather(coords, dim=-1, index=sample_index.unsqueeze(1).repeat(1, 21, 3, 1))

        dist = kernel_size - heatmap_mask * kernel_size
        joint = torch.sum((offset_mask * dist.unsqueeze(2).repeat(1,1,3,1) + coords) * normal_weigth.unsqueeze(2).repeat(1,1,3,1), dim=-1) / normal_weigth.unsqueeze(2).repeat(1,1,3,1).sum(-1)
        return joint


    # change to with weight 2020/4/14
    def offset2joint_weight_nosoftmax(self, offset, depth, kernel_size):
        device = offset.device
        batch_size, joint_num, feature_size, feature_size = offset.size()
        joint_num = int(joint_num / 5)
        if depth.size(-1) != feature_size:
            depth = F.interpolate(depth, size=[feature_size, feature_size])
        offset_unit = offset[:, :joint_num*3, :, :].contiguous()
        heatmap = offset[:, joint_num * 3:joint_num * 4, :, :].contiguous()
        weight = offset[:, joint_num * 4:, :, :].contiguous()
        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
        coords = torch.cat((coords, depth), dim=1).repeat(1, joint_num, 1, 1).view(batch_size, joint_num, 3, -1)
        mask = depth.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        offset_mask = (offset_unit*mask).view(batch_size,joint_num,3,-1)
        heatmap_mask = (heatmap*mask).view(batch_size, joint_num, -1)
        weight_mask = (weight*mask).view(batch_size, joint_num, -1)

        dist = kernel_size - heatmap_mask * kernel_size
        joint = torch.sum((offset_mask * dist.unsqueeze(2).repeat(1, 1, 3, 1) + coords) * weight_mask.unsqueeze(2).repeat(1,1,3,1), dim=-1) / torch.sum(weight_mask, -1,keepdim=True)
        return joint

    def heatmap2joint_softmax(self, heatmap):
        device = heatmap.device
        batch_size, joint_num, feature_size, _= heatmap.size()

        # mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float()+0.5) / feature_size - 1.0
        # mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float()+0.5) / feature_size - 1.0
        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = coords.view(1, 1, 2, -1).repeat(batch_size, joint_num, 1, 1).float().to(device)

        normal_heatmap = F.softmax(heatmap.view(batch_size, joint_num, -1)*30, dim=-1).unsqueeze(2).repeat(1,1,2,1).float()
        out = normal_heatmap * coords.view(batch_size, joint_num, 2, -1)
        joint = torch.sum(out, -1)
        return joint
        # heatmap_temp = heatmap.view(batch_size, joint_num, -1).unsqueeze(2).repeat(1,1,2,1)
        # out =  heatmap_temp * coords.view(batch_size, joint_num, 2, -1)
        # joint = torch.sum(out, -1) / heatmap_temp.sum(-1)
        # return joint

    def plainoffset2joint_softmax(self, offset, weight, kernel_size):
        device = offset.device
        batch_size, joint_num, feature_size,feature_size = offset.size()
        joint_num = int(joint_num / 2)
        # mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        # mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).repeat(1, joint_num, 1, 1).view(batch_size, joint_num, 2, -1).to(device)
        dist = kernel_size - weight * kernel_size
        dist = dist.view(batch_size, joint_num, -1).unsqueeze(2).repeat(1, 1, 2, 1)
        normal_weight = F.softmax(30*weight.view(batch_size, joint_num, -1), dim=-1)
        joint = torch.sum((offset.view(batch_size, joint_num, 2, -1) * dist + coords) * normal_weight.unsqueeze(2).repeat(1,1,2,1), dim=-1)
        return joint

    def weight_pos2joint(self, weight_pos):
        batch_size,joint_num,feature_size,feature_size = weight_pos.size()
        joint_num = int(joint_num / 4)
        weight = F.softmax(weight_pos[:, :joint_num, :, :].contiguous().view(batch_size, joint_num, 1, -1),dim=-1).repeat(1,1,3,1)
        pos = weight_pos[:, joint_num:, :, :].view(batch_size,joint_num,3,-1).contiguous()
        joint = torch.sum(weight * pos, -1)
        return joint.view(-1, joint_num, 3)

    def heatmap_depth2joint(self, pixel_pd, img):
        batch_size,joint_num, feature_size, _ = pixel_pd.size()
        batch_size,joint_num, feature_size, _ = pixel_pd.size()
        img_down = F.interpolate(img, size=[feature_size, feature_size])
        joint_num = joint_num//2
        heatmap, depth = pixel_pd[:, joint_num:, :, :].contiguous(), pixel_pd[:, :joint_num,:,:].contiguous()
        joint_uv = self.heatmap2joint_softmax(heatmap)

        mask = heatmap.ge(0.01).float() * img_down.lt(0.99).float().view(batch_size, 1, feature_size, feature_size)
        heatmap_mask = heatmap * mask
        heatmap_mask = heatmap_mask.view(batch_size, joint_num, -1)

        normal_heatmap = F.softmax(10*heatmap_mask.view(batch_size,joint_num,-1), dim=-1)
        joint_depth = torch.sum(depth.view(batch_size, joint_num, -1) * normal_heatmap, dim=-1).unsqueeze(-1)
        joint = torch.cat((joint_uv, joint_depth), dim=-1)
        return joint
        # joint_depth = torch.sum(depth.view(batch_size, joint_num, -1) * heatmap_mask, dim=-1)/ torch.sum(heatmap_mask, -1)
        # joint = torch.cat((joint_uv, joint_depth.unsqueeze(-1)), dim=-1)
        # return joint

    def heatmap_depthoffset2joint(self, pixel_pd, img):
        batch_size, joint_num, feature_size, _ = pixel_pd.size()
        img_down = F.interpolate(img, size=[feature_size, feature_size])
        joint_num = joint_num//2
        heatmap, depth_offset = pixel_pd[:, joint_num:,:,:].contiguous(), pixel_pd[:, :joint_num,:,:].contiguous()
        joint_uv = self.heatmap2joint_softmax(heatmap)
        depth = img_down + depth_offset
        mask = heatmap.ge(0).float() * img_down.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        heatmap_temp = heatmap.view(batch_size, joint_num, -1) * mask.view(batch_size, joint_num, -1) + 1e-12
        depth = depth * mask
        # depth = heatmap_temp * depth.view(batch_size, joint_num, -1)
        # joint_depth = (torch.sum(depth, -1) / heatmap_temp.sum(-1)).unsqueeze(-1)
        normal_heatmap = F.softmax(heatmap_temp.view(batch_size, joint_num, -1) * 30, dim=-1).float()
        joint_depth = torch.sum(depth.view(batch_size, joint_num, -1) * normal_heatmap, dim=-1).unsqueeze(-1)
        joint = torch.cat((joint_uv, joint_depth), dim=-1)

        return joint

    # change to with mask 2019/9/3
    def plainoffset_depth2joint(self, img, pixel_pd, kernel_size):
        batch_size, joint_num, feature_size, _ = pixel_pd.size()
        joint_num = joint_num // 4
        img_down = F.interpolate(img, size=[feature_size, feature_size])
        offset, weight, depth = pixel_pd[:,:2*joint_num,:,:].contiguous(),pixel_pd[:,2*joint_num:3*joint_num,:,:].contiguous(),\
                              pixel_pd[:,3*joint_num:,:,:].contiguous()
        mask = img_down.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        offset_mask, weight_mask, depth_mask = offset * mask, weight * mask, depth * mask
        joint_uv = self.plainoffset2joint_softmax(offset_mask, weight_mask, kernel_size)
        normal_heatmap = F.softmax(30 * weight_mask.view(batch_size, joint_num, -1), dim=-1)
        joint_depth = torch.sum(depth_mask.view(batch_size, joint_num, -1) * normal_heatmap, dim=-1).unsqueeze(-1)
        joint = torch.cat((joint_uv, joint_depth), dim=-1)
        return joint

    # change to with mask 2019/9/3
    def plainoffset_depthoffset2joint(self, img, pixel_pd, kernel_size):
        batch_size, joint_num,feature_size,_ = pixel_pd.size()
        img_down = F.interpolate(img, size=[feature_size, feature_size])
        joint_num = joint_num // 4
        offset,weight,depth_offset = pixel_pd[:,:2*joint_num,:,:].contiguous(),pixel_pd[:,2*joint_num:3*joint_num,:,:].contiguous(),\
                              pixel_pd[:,3*joint_num:,:,:].contiguous()
        depth = depth_offset + img_down

        mask = img_down.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
        offset_mask, weight_mask, depth_mask = offset * mask, weight * mask, depth * mask

        joint_uv = self.plainoffset2joint_softmax(offset_mask, weight_mask, kernel_size)
        normal_heatmap = F.softmax(30*weight_mask.view(batch_size,joint_num,-1), dim=-1)
        joint_depth = torch.sum(depth_mask.view(batch_size, joint_num, -1) * normal_heatmap, dim=-1).unsqueeze(-1)
        joint = torch.cat((joint_uv, joint_depth), dim=-1)
        return joint

    # output size : B*4(xyz+point_type)*N
    def joint2pc(self, joint, seed = 12345, sample_point=1024, radius=0.08):
        device = joint.device
        batch_size, joint_num, _ = joint.size()

        radius = torch.rand([batch_size, joint_num, 100]).to(device)*radius
        theta = torch.rand([batch_size, joint_num, 100]).to(device)*np.pi
        phi = torch.rand([batch_size, joint_num, 100]).to(device)*np.pi

        x = radius * torch.sin(phi) * torch.cos(theta)
        y = radius * torch.sin(phi) * torch.sin(theta)
        z = radius * torch.cos(phi)
        type = torch.arange(1, joint_num+1).float().to(device).view(1, joint_num, 1).repeat(batch_size, 1, 100)

        point = joint.unsqueeze(-2).repeat(1,1,100,1) + torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)), dim = -1)
        point = torch.cat((point, type.unsqueeze(-1)), dim=-1)
        point = point.view(batch_size,-1,4)
        sample = np.random.choice(point.size(1), sample_point, replace=False)
        return point[:, sample, :].permute(0, 2, 1)

    # depth[-1,1]
    def depth2map(self, depth, heatmap_size=32):
        batchsize, jointnum = depth.size()
        depthmap = ((depth + 1) / 2).contiguous().view(batchsize, jointnum, 1, 1).expand(batchsize, jointnum, heatmap_size, heatmap_size)
        return depthmap

    # select feature
    def joint2feature(self, joint, img, feature_paras, feature_size, feature_types):
        device = img.device
        all_feature = torch.Tensor().to(device)
        batch_size, joint_num, _ = joint.size()
        for feature_index, feature_type in enumerate(feature_types):
            if 'heatmap' == feature_type:
                heatmap, mask = self.joint2heatmap2d(joint, img, feature_paras[feature_index], feature_size)
                feature = heatmap
            elif 'heatmap_depth' == feature_type:
                heatmap, mask = self.joint2heatmap2d(joint, img, feature_paras[feature_index], feature_size)
                depth = torch.ones_like(heatmap).to(device) * joint[:, :, 2].view(joint.size(0), -1, 1, 1)
                depth[mask == 0] = -1
                feature = torch.cat((heatmap, depth), dim=1)
            elif 'heatmap_depthoffset' == feature_type:
                heatmap, mask = self.joint2heatmap2d(joint, img, feature_paras[feature_index], feature_size)
                depth_offset = self.joint2depthoffset(joint, img, mask, feature_size)
                feature = torch.cat((heatmap, depth_offset), dim=1)
            elif 'plainoffset_depth' == feature_type:
                plain_offset, mask = self.joint2plainoffset(joint, img, feature_paras[feature_index], feature_size)
                depth = torch.ones([joint.size(0),joint.size(1),feature_size,feature_size]).to(device) * joint[:, :, 2].view(joint.size(0), -1, 1, 1)
                feature = torch.cat((plain_offset, depth), dim=1)
            elif 'plainoffset_depthoffset' == feature_type:
                plain_offset, mask = self.joint2plainoffset(joint, img, feature_paras[feature_index], feature_size)
                depth_offset = self.joint2depthoffset(joint, img, mask, feature_size)
                feature = torch.cat((plain_offset, depth_offset), dim=1)
            elif 'offset' == feature_type or 'weight_offset' == feature_type or 'weight_offset_nosoftmax' == feature_type:
                feature = self.joint2offset(joint, img, feature_paras[feature_index], feature_size)
            elif feature_type == 'weight_pos':
                feature = joint.view(batch_size,joint_num,3,1,1).repeat(1,1,1,feature_size,feature_size)
                feature = feature.view(batch_size,-1,feature_size,feature_size)
            # elif feature_type == 'weight_2.5d_heatmap':
            #     raise NotImplementedError()
            all_feature = torch.cat((all_feature, feature), dim=1)
        return all_feature

    # predict joint from pixel-wise estimation
    def feature2joint(self, img, pixel_pd, feature_types, feature_paras):
        for feature_index, feature_type in enumerate(feature_types):
            if feature_type == 'heatmap':
                device = img.device
                joint_uv = self.heatmap2joint_softmax(pixel_pd)
                joint_d = torch.zeros(joint_uv.size(0),joint_uv.size(1),1).to(device)
                joint = torch.cat((joint_uv,joint_d),dim=-1)
            elif feature_type == 'offset':
                # joint = self.offset2joint(pixel_pd, img, feature_paras[feature_index])
                joint = self.offset2joint_softmax(pixel_pd, img, feature_paras[feature_index])
            elif feature_type == 'weight_offset':
                joint = self.offset2joint_weight(pixel_pd, img, feature_paras[feature_index])
                # joint = self.offset2joint_weight_sample(pixel_pd, img, feature_paras[feature_index])
            elif feature_type == 'weight_offset_nosoftmax':
                joint = self.offset2joint_weight_nosoftmax(pixel_pd, img, feature_paras[feature_index])
            elif feature_type == 'heatmap_depth':
                joint = self.heatmap_depth2joint(pixel_pd, img)
            elif feature_type == 'heatmap_depthoffset':
                joint = self.heatmap_depthoffset2joint(pixel_pd, img)
            elif feature_type == 'plainoffset_depth':
                joint = self.plainoffset_depth2joint(img, pixel_pd, feature_paras[feature_index])
            elif feature_type == 'plainoffset_depthoffset':
                joint = self.plainoffset_depthoffset2joint(img, pixel_pd, feature_paras[feature_index])
            elif feature_type == 'weight_pos':
                joint = self.weight_pos2joint(pixel_pd)
            # elif feature_type == 'weight_2.5d_heatmap':
            #     joint = self.latent_heatmap2joint_weight(pixel_pd)

        return joint


    def pcl_joint2offset(self, joint, pcl, kernel_size):
        """
        :param: joint BxJx3
        :param: pcl BxNx3
        """
        B, J, _ = joint.size()
        N = pcl.size(1)
        device = joint.device
        offset = joint.unsqueeze(2) - pcl.unsqueeze(1)
        dis = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=-1))
        offset_norm = offset / (dis.unsqueeze(-1) + 1e-8)
        offset_norm = offset_norm.permute(0, 1, 3, 2).reshape(B, J, 3, N)

        if torch.is_tensor(kernel_size):
            kernel_size = kernel_size.to(device)
            dis = (kernel_size.view(1, J, 1) - dis) / kernel_size.view(1, J, 1)
        else:
            dis = (kernel_size - dis) / kernel_size

        mask = dis.ge(0).float() * pcl[:, :, 2:].lt(0.99).float().permute(0, 2, 1)
        dis = dis * mask
        offset_norm = offset_norm * mask.view(B, J, 1, N)
        offset_norm = offset_norm.reshape(B, J*3, N)
        return torch.cat((offset_norm, dis), dim=1).to(device).permute(0, 2, 1)

    def pcl_offset2joint_weight(self, pcl_result, pcl, kernel_size):
        """
        :param: pcl_result BxNx(5*J)
        :param: pcl BxNx3
        """
        assert pcl.size(2) == 3
        device = pcl.device
        pcl_result = pcl_result.permute(0, 2, 1)
        B, J, N = pcl_result.size()
        J = int(J / 5)

        coords = pcl.permute(0, 2, 1).reshape(B, 1, 3, N)
        offset = pcl_result[:, :J * 3, :].view(B, J, 3, N)
        heatmap = pcl_result[:, J * 3:J * 4, :].view(B, J, 1, N)
        weight = pcl_result[:, -J:, :].view(B, J, 1, N)

        mask = pcl[:, :, 2].gt(0.99).view(B, 1, 1, N)
        weight_mask = torch.masked_fill(weight, mask, -1e8)
        normal_weight = F.softmax(weight_mask, dim=-1)

        if torch.is_tensor(kernel_size):
            kernel_size = kernel_size.to(device)
            dist = kernel_size.view(1, J, 1, 1) - heatmap * kernel_size.view(1, J, 1, 1)
        else:
            dist = kernel_size - heatmap * kernel_size

        joint = torch.sum((offset * dist + coords) * normal_weight, dim=-1)
        return joint

    def img_pcl(self, img, S):
        device = img.device
        B = img.size(0)
        img = F.interpolate(img, size=[S, S])
        mesh_x = 2.0 * (torch.arange(S).unsqueeze(1).expand(S,S).float() + 0.5) / S - 1.0
        mesh_y = 2.0 * (torch.arange(S).unsqueeze(0).expand(S,S).float() + 0.5) / S - 1.0
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(B, 1, 1, 1).to(device)
        pcl = torch.cat((coords, img), dim=1).view(B,3,S*S).permute(0, 2, 1)
        return pcl

    # lxy
    def offset2joint_weight_2D(self, offset, kernel_size=None, hand_mask=None,joint_nb=21, beta=15):
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
        weight = self.softmax(weight * beta)
        # mask = dis.gt(0).unsqueeze(2)
        dis = kernel_size - dis * kernel_size
        if hand_mask is not None:
            hand_mask = hand_mask.reshape(B, J, 1, -1)  # B J 1 S S
            joint = torch.sum((offset_unit * dis.unsqueeze(2).repeat(1, 1, 2, 1)*hand_mask + coords*hand_mask) *
                              weight.unsqueeze(2).repeat(1, 1, 2, 1), dim=-1)
        else:
            joint = torch.sum((offset_unit * dis.unsqueeze(2).repeat(1, 1, 2, 1) + coords)*weight.unsqueeze(2).repeat(1, 1, 2, 1), dim=-1)
        # joint = torch.sum((offset_unit * dis.unsqueeze(2).repeat(1, 1, 2, 1) * mask + coords * mask), dim=-1) /\
        #                   torch.sum(mask, dim=-1)
        return joint

    def joint2offset_weight_2D(self, joint, offset_size, kernel_size=None, hand_mask=None,joint_nb=21):
        device = joint.device
        B, J, _ = joint.size()
        J = joint_nb
        if kernel_size is None:
            kernel_size = kernel_size

        joint_feature = joint.reshape(B, -1, 1, 1).repeat(1, 1, offset_size, offset_size)
        mesh_x = (torch.arange(offset_size).unsqueeze(1).expand(offset_size, offset_size).float() + 0.5) / offset_size
        mesh_y = (torch.arange(offset_size).unsqueeze(0).expand(offset_size, offset_size).float() + 0.5) / offset_size
        coords = torch.stack((mesh_y, mesh_x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(B, J, 1, 1).to(device)
        offset = joint_feature - coords
        offset = offset.view(B, J, 2, offset_size, offset_size)
        dist = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=2) + 1e-8)
        offset_norm = (offset / (dist.unsqueeze(2)))    # B J 1 S S
        heatmap = (kernel_size - dist) / kernel_size    # B J S S
        mask = heatmap.ge(0).float()
        # hand_mask = hand_mask.unsqueeze(1).repeat(1,J,1,1).float()  # B J S S
        if hand_mask is not None:
            offset_norm_mask = (offset_norm * mask.unsqueeze(2) * hand_mask.unsqueeze(2)).view(B, -1, offset_size, offset_size).float()
            heatmap_mask = heatmap * mask.float() * hand_mask
        else:
            offset_norm_mask = (offset_norm * mask.unsqueeze(2)).view(B, -1, offset_size, offset_size).float()
            heatmap_mask = heatmap * mask.float()
        return torch.cat((offset_norm_mask, heatmap_mask), dim=1)

    def joint2heatmap(self, joint, std, heatmap_size,sigma = 1.5):
        # joint depth is norm[-1,1]
        divce = joint.device
        #img_down = F.interpolate(img, size=[heatmap_size, heatmap_size])
        batch_size, joint_num, _ = joint.size()
        xx, yy = np.meshgrid(np.arange(heatmap_size), np.arange(heatmap_size))
        mesh_x = torch.from_numpy(xx).view(1, 1, heatmap_size, heatmap_size).repeat(batch_size, joint_num, 1, 1).float().to(divce)+0.5
        mesh_y = torch.from_numpy(yy).view(1, 1, heatmap_size, heatmap_size).repeat(batch_size, joint_num, 1, 1).float().to(divce)+0.5
        joint_ht = torch.zeros_like(joint).to(divce)
        joint_ht[:, :, 0] = (joint[:, :, 0] + 1) / 2 * heatmap_size
        joint_ht[:, :, 1] = (joint[:, :, 1] + 1) / 2 * heatmap_size
        joint_x = joint_ht[:, :, 0].view(batch_size, joint_num, 1, 1).repeat(1, 1, heatmap_size, heatmap_size).float()
        joint_y = joint_ht[:, :, 1].view(batch_size, joint_num, 1, 1).repeat(1, 1, heatmap_size, heatmap_size).float()
        heatmap = torch.exp(-(torch.pow((mesh_x - joint_x) / std, 2) + torch.pow((mesh_y.to(divce) - joint_y) / std, 2))/(2*pow(sigma,2)))
        #heatmap = torch.exp(-(torch.pow((mesh_x - joint_x) / std, 2) + torch.pow((mesh_y.to(divce) - joint_y) / std, 2)))
        #mask = heatmap.ge(0.01).float() * img_down.lt(0.99).float().view(batch_size, 1, heatmap_size, heatmap_size)
        return heatmap


    def joint2heatmap_skeleton(self, joint, std, heatmap_size,sigma = 1):
        skelen2joint = [
                [0, 13], [13, 14], [14, 15], [15, 20],
                [0, 1], [1, 2], [2, 3], [3, 16],
                [0, 4], [4, 5], [5, 6], [6, 17],
                [0, 10], [10, 11], [11,  12], [12, 19],
                [0, 7], [7, 8], [8, 9], [9, 18]
                ]
        # joint depth is norm[-1,1]
        divce = joint.device
        #img_down = F.interpolate(img, size=[heatmap_size, heatmap_size])
        batch_size, joint_num, _ = joint.size()
        xx, yy = np.meshgrid(np.arange(heatmap_size), np.arange(heatmap_size))
        mesh_x = torch.from_numpy(xx).view(1, 1, heatmap_size, heatmap_size).repeat(batch_size, joint_num-1, 1, 1).float().to(divce)+0.5
        mesh_y = torch.from_numpy(yy).view(1, 1, heatmap_size, heatmap_size).repeat(batch_size, joint_num-1, 1, 1).float().to(divce)+0.5
        joint_ht = torch.zeros_like(joint).to(divce)
        joint_ht[:, :, 0] = (joint[:, :, 0] + 1) / 2 * heatmap_size
        joint_ht[:, :, 1] = (joint[:, :, 1] + 1) / 2 * heatmap_size
        joint_x = joint_ht[:, :, 0].view(batch_size, joint_num, 1, 1).repeat(1, 1, heatmap_size, heatmap_size).float()
        joint_y = joint_ht[:, :, 1].view(batch_size, joint_num, 1, 1).repeat(1, 1, heatmap_size, heatmap_size).float()
        heatmap = torch.exp(-(torch.pow((mesh_x - joint_x[:,1:]) / std, 2) + torch.pow((mesh_y.to(divce) - joint_y[:,1:]) / std, 2))/(2*pow(sigma,2)))

        for i in range(joint_num-1):
            joint1_x = joint_ht[:, skelen2joint[i][0], 0].view(batch_size, 1,  1).repeat(1,  heatmap_size,heatmap_size).float()
            joint1_y = joint_ht[:, skelen2joint[i][0], 1].view(batch_size, 1,  1).repeat(1,  heatmap_size,heatmap_size).float()
            joint2_x = joint_ht[:, skelen2joint[i][1], 0].view(batch_size, 1,  1).repeat(1,  heatmap_size, heatmap_size).float()
            joint2_y = joint_ht[:, skelen2joint[i][1], 1].view(batch_size, 1,  1).repeat(1,  heatmap_size, heatmap_size).float()

            # heatmap = torch.exp(
            #     -(torch.pow((mesh_x - joint_x) / std, 2) + torch.pow((mesh_y.to(divce) - joint_y) / std, 2)) / (
            #                 2 * pow(sigma, 2)))

            # heatmap[:,i] = torch.exp(-(
            #     (torch.sqrt((torch.pow((mesh_x[:,i] - joint1_x) / std, 2)) + torch.pow((mesh_y[:,i] .to(divce) - joint1_y) / std, 2))+
            #     torch.sqrt((torch.pow((mesh_x[:,i]  - joint2_x) / std, 2)) + torch.pow((mesh_y[:,i] .to(divce) - joint2_y) / std, 2))) / (
            #     #2 * sqrt(pow(1-pow(sigma, 2),2))
            #     2*pow(sigma, 2)
            # )
            # ))

            # distance to start keypoints
            d2_start = ((mesh_x[:,i] - joint1_x) ** 2 + (mesh_y[:,i] - joint1_y) ** 2)

            # distance to end keypoints
            d2_end = ((mesh_x[:, i] - joint2_x) ** 2 + (mesh_y[:, i] - joint2_y) ** 2)

            # the distance between start and end keypoints.
            d2_ab = ((joint1_x - joint2_x) ** 2 + (joint1_y - joint2_y) ** 2)

            # if d2_ab < 1:
            #     self.generate_a_heatmap(arr, start[None], start_value[None])
            #     continue

            coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate.int() - b_dominate.int()

            position = torch.stack([mesh_x[:,i], mesh_y[:,i]], dim=-1)
            projection = torch.stack([joint1_x, joint1_y], dim=-1) + torch.stack([coeff, coeff], dim=-1) * (torch.stack([joint2_x, joint2_y], dim=-1) - torch.stack([joint1_x, joint1_y], dim=-1))
            d2_line = position - projection
            d2_line = d2_line[:, :, :,0] ** 2 + d2_line[:, :, :,1] ** 2
            d2_seg = a_dominate.int() * d2_start + b_dominate.int() * d2_end + seg_dominate * d2_line
            if skelen2joint[i][0]==0:
                heatmap[:, i] = torch.exp(-d2_seg / 2. / sigma ** 2)/2
            else:
                heatmap[:,i] = torch.exp(-d2_seg / 2. / sigma ** 2)
        # heatmap[:, torch.arange(joint_num-1)] = torch.exp(-(
        #         (torch.sqrt((torch.pow((mesh_x - joint1_x) / std, 2)) + torch.pow(
        #             (mesh_y[:, i].to(divce) - joint1_y) / std, 2)) +
        #          torch.sqrt((torch.pow((mesh_x[:, i] - joint2_x) / std, 2)) + torch.pow(
        #              (mesh_y[:, i].to(divce) - joint2_y) / std, 2))) / (
        #                 2 * sqrt(pow(1 - pow(sigma, 2), 2))
        #         )

        return heatmap

    def rigid_transform_3D(self,A, B):
        n, dim = A.shape
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
        U, s, V = np.linalg.svd(H)
        R = np.dot(np.transpose(V), np.transpose(U))
        if np.linalg.det(R) < 0:
            s[-1] = -s[-1]
            V[2] = -V[2]
            R = np.dot(np.transpose(V), np.transpose(U))

        varP = np.var(A, axis=0).sum()
        c = 1 / varP * np.sum(s)

        t = -np.dot(c * R, np.transpose(centroid_A)) + np.transpose(centroid_B)
        return c, R, t

    def rigid_align(self,A, B):

        c, R, t = self.rigid_transform_3D(A, B)
        A2 = np.transpose(np.dot(c * R, np.transpose(A))) + t
        return A2
