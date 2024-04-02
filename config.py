import os.path as osp
JOINT = {
    'STB': 21,
    'nyu':14,
    'dexycb':21,
    'ho3d':21
}

STEP = {
    'STB': 20,
    'nyu': 25,
    'dexycb':10,
    'ho3d':19
}

EPOCH = {
    'STB': 30,
    'nyu': 30,
    'dexycb':15,
    'ho3d':24
}

CUBE = {
    'STB': [200,200,200],
    'nyu': [250, 250, 250],
    'dexycb':[250, 250, 250],
    'ho3d': [280, 280, 280],
}


class Config(object):
    phase = 'test'
    root_dir = '/home/cyc/pycharm/data/hand/' # set dataset root path

    net = 'KPFusion-resnet-18' #['KPFusion-resnet-18', 'KPFusion-convnext-T']

    dataset = 'dexycb'  # ['nyu','STB', 'dexycb', 'ho3d']
    ho3d_version = 'v2'
    model_save = ''
    save_dir = './'
    dexycb_setup = 's0'
    pretrain = '1k'
    point_num = 1024

    load_model = './checkpoint/dexycb/KPFusion_Dexycb_s0.pth'
    finetune_dir = ''

    gpu_id = '0'

    joint_num = JOINT[dataset]

    batch_size = 64
    input_size = 128
    cube_size = CUBE[dataset]
    center_type = 'refine'
    loss_type = 'L1Loss'  # ['L1Loss', 'Mse','GHM']
    augment_para = [10, 0.2, 180]
    color_factor = 0.2

    lr = 8e-4
    max_epoch = EPOCH[dataset]
    step_size = STEP[dataset]
    opt = 'adamw'  # ['sgd', 'adam']
    scheduler = 'step'  # ['auto', 'step', 'constant']
    downsample = 2 # [1,2,4,8]

    awr = True
    coord_weight = 100
    deconv_weight = 1
    spatial_weight = [10,10,10]
    spatial_epoch = [24, 24, 24]

    # for AWR backbone
    feature_type = ['weight_offset']  #['weight_offset', 'weight_pos','heatmap_depthoffset','plainoffset_depth','plainoffset_depthoffset', 'offset']
    feature_para = [0.8]

    stage_type = [1,1,2,3,2,3]  # Depth backbone, RGB backbone, (RGB KFAM, Depth KFAM,) (RGB KFAM, Depth KFAM)

    mano_path = osp.join('./util', 'manopth')


opt = Config()
