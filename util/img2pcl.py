import math

import torch


class Pcl_utils(object):
    def __init__(self):
        self.flip = 1
        self.sample_num = 1024

    def getpcl(self,imgD,com3D, cube,M, cam_para):
        B = imgD.shape[0]
        imgD = imgD.squeeze(1)#B*128*128
        mask = torch.isclose(imgD, torch.tensor(1.))
        dpt_ori = imgD * (cube[:,2] / 2.0).unsqueeze(1).unsqueeze(1) + com3D[:,2].unsqueeze(1).unsqueeze(1)
        # change the background value
        dpt_ori[mask] = 0

        pcl = self.depthTopcl(dpt_ori, M, cam_para) - com3D
        pcl_num = pcl.shape[1]
        cube_tile = (cube / 2.0).repeat(1,pcl_num).reshape([B,pcl_num, 3])
        pcl = pcl / cube_tile

        pcl_index = torch.arange(pcl.shape[1]).unsqueeze(0).repeat(B,-1)
        pcl_num = pcl.shape[1]
        if pcl_num == 0:
            pcl_sample = torch.zeros([B,self.sample_num, 3])
        else:
            if pcl_num < self.sample_num:
                tmp = math.floor(self.sample_num / pcl_num)
                index_temp = pcl_index.repeat(1,tmp+1)
                pcl_index = torch.cat([index_temp, torch.multinomial(pcl_index, divmod(self.sample_num, pcl_num)[1], replacement=False)],dim=-1)
            select = torch.multinomial(pcl_index, self.sample_num, replacement=False)
            pcl_sample = pcl[select, :]
        #select = np.random.choice(pcl_index, self.sample_num, replace=False)
        #select = torch.randint(pcl_index,self.sample_num,)
        # select = torch.multinomial(pcl_index,self.sample_num,replacement=False)
        # pcl_sample = pcl[select, :]

        return pcl_sample

    def depthTopcl(self,dpt, T, paras, background_val=torch.tensor(0.)):
        fx, fy, fu, fv = paras[:,0], paras[:,1], paras[:,2], paras[:,3]
        # get valid points and transform

        pts = torch.nonzero(~torch.isclose(dpt, background_val))#N*3

        #pts = torch.cat([pts[:,:, [1, 0]] + 0.5, torch.ones((pts.shape[1], 1), dtype=torch.float32)], dim=2)
        pts = torch.cat([pts[:, [2, 1]] + 0.5, torch.ones((pts.shape[0], 1), dtype=torch.float32)], dim=1)
        print("torch.linalg.inv(T)",torch.linalg.inv(T).shape)
        print("pts.T", pts.T.shape)

        pts = torch.bmm(torch.linalg.inv(T), pts[:,1:].transpose(1,2)).transpose(1,2)
        pts = (pts[:,:, 0:2] / pts[:,:, 2].unsqueeze(2)).reshape((pts.shape[0],pts.shape[1], 2))

        # replace the invalid data
        depth = dpt[(~torch.isclose(dpt, background_val))]

        # get x and y data in a vectorized way
        row = (pts[:,:, 0] - fu) / fx * depth
        col = self.flip * (pts[:,:, 1] - fv) / fy * depth

        # combine x,y,depth
        return torch.stack((row, col, depth),dim=-1)

if __name__ == "__main__":
    img = torch.randn(64,1,128,128)
    com3D = torch.zeros(64,3)
    cube = torch.ones(64,3)
    M= torch.rand(64,3,3)
    paras = torch.rand(64, 4)
    pcl_update = Pcl_utils().getpcl(img,com3D, cube,M, paras)