import numpy as np
import cv2
import random
from config import opt
import json
import os
from PIL import Image
import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cv2pil(cv_img):
    return Image.fromarray(cv2.cvtColor(np.uint8(cv_img), cv2.COLOR_BGR2RGB))

def uvd2xyz(uvd, K):
    fx, fy, fu, fv = K[0, 0], K[0, 0], K[0, 2], K[1, 2]
    xyz = np.zeros_like(uvd, np.float32)
    xyz[:, 0] = (uvd[:, 0] - fu) * uvd[:, 2] / fx
    xyz[:, 1] = (uvd[:, 1] - fv) * uvd[:, 2] / fy
    xyz[:, 2] = uvd[:, 2]
    return xyz

def xyz2uvd(xyz, K):
    fx, fy, fu, fv = K[0, 0], K[0, 0], K[0, 2], K[1, 2]
    uvd = np.zeros_like(xyz, np.float32)
    uvd[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
    uvd[:, 1] = (xyz[:, 1] * fy / xyz[:, 2] + fv)
    uvd[:, 2] = xyz[:, 2]
    return uvd

def xyz2uvd_torch(xyz, K):
    fx, fy, fu, fv = K[0, 0], K[0, 0], K[0, 2], K[1, 2]
    uvd = torch.zeros_like(xyz).to(device)
    uvd[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
    uvd[:, 1] = (xyz[:, 1] * fy / xyz[:, 2] + fv)
    uvd[:, 2] = xyz[:, 2]
    return uvd

def STB_xyz2uvd(xyz, K, rotationMatrix, T):
    '''
    :param xyz: (21, 3)
    :param K: (3, 4)
    :return:
    '''
    uvd = np.dot(K, np.dot(np.transpose(rotationMatrix), xyz.transpose(1, 0) - T)).transpose(1, 0)
    uvd = uvd / uvd[:, 2:3]
    return np.concatenate((uvd[:, :2], xyz[:, 2:3]), axis=1)

""" General util functions. """
def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg


def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

""" Dataset related functions. """
def db_size(set_name):
    """ Hardcoded size of the datasets. """
    if set_name == 'training':
        return 32560  # number of unique samples (they exists in multiple 'versions')
    elif set_name == 'evaluation':
        return 3960
    else:
        assert 0, 'Invalid choice.'

# Mesh
def load_db_annotation(base_path, set_name=None):
    if set_name is None:
        # only training set annotations are released so this is a valid default choice
        set_name = 'training'

    print('Loading FreiHAND dataset index ...')
    t = time.time()

    # assumed paths to data containers
    k_path = os.path.join(base_path, '%s_K.json' % set_name)
    xyz_path = os.path.join(base_path, '%s_xyz.json' % set_name)
    scale_path = os.path.join(base_path, '%s_scale.json' % set_name)
    vert_path = os.path.join(base_path, '%s_verts.json' % set_name)

    # load if exist
    K_list = json_load(k_path)
    vert_list = json_load(vert_path)
    xyz_list = json_load(xyz_path)
    scale_list = json_load(scale_path)

    # should have all the same length
    assert len(K_list) == len(vert_list), 'Size mismatch.'
    assert len(K_list) == len(xyz_list), 'Size mismatch.'
    assert len(K_list) == len(scale_list), 'Size mismatch.'

    print('Loading of %d samples done in %.2f seconds' % (len(K_list), time.time()-t))
    return list(zip(K_list, vert_list, xyz_list, scale_list))

# def load_db_annotation(base_path, set_name=None):
#     if set_name is None:
#         # only training set annotations are released so this is a valid default choice
#         set_name = 'training'
#
#     print('Loading FreiHAND dataset index ...')
#     t = time.time()
#
#     # assumed paths to data containers
#     k_path = os.path.join(base_path, '%s_K.json' % set_name)
#     xyz_path = os.path.join(base_path, '%s_xyz.json' % set_name)
#     scale_path = os.path.join(base_path, '%s_scale.json' % set_name)
#     vert_path = os.path.join(base_path, '%s_verts.json' % set_name)
#
#     # load if exist
#     K_list = json_load(k_path)
#     xyz_list = json_load(xyz_path)
#     scale_list = json_load(scale_path)
#
#     # should have all the same length
#     assert len(K_list) == len(xyz_list), 'Size mismatch.'
#     assert len(K_list) == len(scale_list), 'Size mismatch.'
#
#     print('Loading of %d samples done in %.2f seconds' % (len(K_list), time.time()-t))
#     return list(zip(K_list, xyz_list, scale_list))

class sample_version:
    gs = 'gs'  # green screen
    hom = 'hom'  # homogenized
    sample = 'sample'  # auto colorization with sample points
    auto = 'auto'  # auto colorization without sample points: automatic color hallucination

    db_size = db_size('training')

    @classmethod
    def valid_options(cls):
        return [cls.gs, cls.hom, cls.sample, cls.auto]


    @classmethod
    def check_valid(cls, version):
        msg = 'Invalid choice: "%s" (must be in %s)' % (version, cls.valid_options())
        assert version in cls.valid_options(), msg

    @classmethod
    def map_id(cls, id, version):
        cls.check_valid(version)
        return id + cls.db_size*cls.valid_options().index(version)


def read_img(idx, base_path, set_name, version=None):
    if version is None:
        version = sample_version.gs

    if set_name == 'evaluation':
        assert version == sample_version.gs, 'This the only valid choice for samples from the evaluation split.'

    img_rgb_path = os.path.join(base_path, set_name, 'rgb',
                                '%08d.jpg' % sample_version.map_id(idx, version))
    _assert_exist(img_rgb_path)
    return cv2.imread(img_rgb_path)

def imcrop(img, center, crop_size):
    x1 = int(np.round(center[0]-crop_size))
    y1 = int(np.round(center[1]-crop_size))
    x2 = int(np.round(center[0]+crop_size))
    y2 = int(np.round(center[1]+crop_size))

    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
         img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)

    if img.ndim < 3: # for depth
        img_crop = img[y1:y2, x1:x2]
    else: # for rgb
        img_crop = img[y1:y2, x1:x2, :]

    trans = np.eye(3)
    trans[0, 2] = -x1
    trans[1, 2] = -y1

    return img_crop, trans

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    borderValue = [127, 127, 127]

    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                                 -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_CONSTANT, value=borderValue)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2

def convert_kp(keypoints):
    kp_dict = {0: 0, 1: 20, 2: 19, 3: 18, 4: 17, 5: 16, 6: 15, 7: 14, 8: 13, 9: 12, 10: 11, 11: 10,
               12: 9, 13: 8, 14: 7, 15: 6, 16: 5, 17: 4, 18: 3, 19: 2, 20: 1}

    keypoints_new = list()
    for i in range(21):
        if i in kp_dict.keys():
            pos = kp_dict[i]
            keypoints_new.append(keypoints[pos, :])

    return np.stack(keypoints_new, 0)

def preprocessRHD(image, mask, kp_coord_uv, kp_coord_xyz):
    # hand side: left
    temp_mask = mask.copy()
    mask = mask[:, :, 0]
    image_size = image.shape[1]
    cond_l = np.logical_and(mask > 1, mask < 18)
    cond_r = mask > 17
    num_px_left_hand = np.sum(cond_l)
    num_px_right_hand = np.sum(cond_r)
    hand_side = num_px_left_hand > num_px_right_hand
    if hand_side:
        xyz = kp_coord_xyz[:21, :]
        uv = kp_coord_uv[:21, :]
    else:
        xyz = kp_coord_xyz[-21:, :]
        uv = kp_coord_uv[-21:, :]

    if not hand_side:
        image = cv2.flip(image, 1)
        temp_mask = cv2.flip(temp_mask, 1)
        mask = cv2.flip(mask, 1)
        xyz[:, 0] = -xyz[:, 0]
        uv[:, 0] = image_size - uv[:, 0]

    # flip left to right
    # image = cv2.flip(image, 1)
    # temp_mask = cv2.flip(temp_mask, 1)
    # mask = cv2.flip(mask, 1)
    # xyz[:, 0] = -xyz[:, 0]
    # uv[:, 0] = image_size - uv[:, 0]

    if hand_side:
        y, x = np.where(np.logical_and(mask > 1, mask < 18))
    else:
        y, x = np.where(mask > 17)

    ratio = 1 / 0.8
    # ratio = 1 / 0.6
    max_x = max(x)
    max_y = max(y)
    min_x = min(x)
    min_y = min(y)
    crop_center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
    crop_size = max((max_x - min_x), (max_y - min_y)) * ratio // 2
    # crop_center = (160, 160)
    # crop_size = 160

    if hand_side:
        hand_side_out = np.array([1.0, 0.0])
    else:
        hand_side_out = np.array([0.0, 1.0])

    return image, temp_mask, xyz, crop_center, crop_size, hand_side_out

def preprocessSTB(uvd):
    uvd = convert_kp(uvd)
    wrist_uvd = uvd[16, :] + 2.0 * (uvd[0, :] - uvd[16, :])
    uvd = np.concatenate([np.expand_dims(wrist_uvd, 0), uvd[1:, :]], 0)
    ratio = 1 / 0.8  # default 1.2
    max = np.max(uvd[:, :2], axis=0, keepdims=True)
    min = np.min(uvd[:, :2], axis=0, keepdims=True)
    crop_center = ((max + min) // 2).reshape(2)
    crop_size = int((np.max(max - min, axis=1) * ratio)//2)

    return uvd, crop_center, crop_size

def process_augmentated_coords(uvd, xyz, img2bb_trans, inv_trans_joint, K):
    fh_order = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17]
    uvd = uvd.copy()
    xyz = xyz.copy()
    uv1 = np.concatenate((uvd[:, :2], np.ones_like(uvd[:, :1])), 1)
    uvd[:, :2] = np.dot(img2bb_trans, uv1.transpose(1, 0)).transpose(1, 0)[:, :2]
    # uvd back to xyz and compute new scale
    uvd_compute_bone = uvd.copy()
    uv1 = np.concatenate((uvd_compute_bone[:, :2], np.ones_like(uvd[:, :1])), 1)
    uvd_compute_bone[:, :2] = np.dot(inv_trans_joint, uv1.transpose(1, 0)).transpose(1, 0)[:, :2]
    xyz_compute_bone = uvd2xyz(uvd_compute_bone, K)
    scale = np.sqrt(np.sum(np.square(xyz_compute_bone[12, :] - xyz_compute_bone[11, :])))
    # create heatmap
    hm_size = 64
    ratio = opt.input_img_shape[0] / hm_size
    uv_for_hm = (uvd[:, :2].copy() // ratio)
    uv_for_hm = uv_for_hm[fh_order]
    heatmap = create_heatmap(uv_for_hm, hm_size, np.sqrt(2.5))
    # normalize coordinates
    root_depth = uvd[12:13, 2:3].copy()
    uvd[:, 2:3] = (uvd[:, 2:3] - root_depth) / scale
    uvd[:, :2] = uvd[:, :2] / (opt.input_img_shape[0] // 2) - 1
    xyz = xyz[fh_order]
    uvd = uvd[fh_order]
    return uvd, xyz, heatmap, root_depth, scale

def create_heatmap(joint, ht_size=64, std = np.sqrt(2.5)):
    '''

    :param joint: absolute locations in uv coordinates [[32, 32], [1, 12]]
    :param ht_size:
    :param std:
    :return:
    '''

    joint_num = joint.shape[0]
    if (joint == [0, 0]).all():
        return np.zeros((joint_num, ht_size, ht_size))

    xx, yy = np.meshgrid(np.arange(ht_size), np.arange(ht_size))

    mesh_x = xx.reshape(1, ht_size, ht_size).repeat(joint_num, axis=0).astype(np.float32)  # shape(21,64,64)
    mesh_y = yy.reshape(1, ht_size, ht_size).repeat(joint_num, axis=0).astype(np.float32)

    joint_x = np.tile(joint[:, 0].reshape(joint_num, 1, 1), (1, ht_size, ht_size))  # shape(21,64,64)
    joint_y = np.tile(joint[:, 1].reshape(joint_num, 1, 1), (1, ht_size, ht_size))  # shape(21,64,64)
    heatmap = np.exp(-(np.power((mesh_x-joint_x)/std, 2) + np.power((mesh_y-joint_y) / std, 2)))
    return heatmap

def unify_mask(cropped_mask, hand_side):
    # test mask
    # image_crop = image_crop / 255.
    # if hand_side[0]:
    #     # [0] background [1] people [2, 3, 4] thumb [5, 6, 7] index
    #     # [8, 9, 10] middle [11, 12, 13] fourth [14, 15, 16] little [17] palm
    #     image_crop[..., 2] = np.where(np.logical_and(mask_crop >= 17, mask_crop < 18),
    #                                   255, image_crop[..., 2])
    # else:
    #     # [18, 19, 20] thumb [21, 22, 23] index [24, 25, 26] middle
    #     # [27, 28, 29] fourth [30, 31, 32] little [33] palm
    #     image_crop[..., 2] = np.where(np.logical_and(mask_crop >= 33, mask_crop < 34), 255, image_crop[..., 2])
    #     cv2.imshow('1', image_crop)
    #     cv2.waitKey()
    if hand_side[0]:
        cropped_mask = np.where(np.logical_and(cropped_mask == 1, cropped_mask >= 18), 0, cropped_mask)
        cropped_mask = np.where(cropped_mask >= 1, cropped_mask - 1, cropped_mask)
    else:
        cropped_mask = np.where(np.logical_and(cropped_mask >= 1, cropped_mask < 18), 0, cropped_mask)
        cropped_mask = np.where(cropped_mask >= 1, cropped_mask - 17, cropped_mask)
    # fix wrong annotations now: [0] background [1-16] hand
    cropped_mask = np.where(cropped_mask >= 17, 0, cropped_mask)

    return cropped_mask

def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order == 'RGB':
        img = img[:, :, ::-1].copy()

    img = img.astype(np.float32)
    return img

def get_bbox(joint_img, joint_valid):
    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    x_img = x_img[joint_valid == 1];
    y_img = y_img[joint_valid == 1];
    xmin = min(x_img);
    ymin = min(y_img);
    xmax = max(x_img);
    ymax = max(y_img);

    x_center = (xmin + xmax) / 2.;
    width = xmax - xmin;
    xmin = x_center - 0.5 * width * 1.2
    xmax = x_center + 0.5 * width * 1.2

    y_center = (ymin + ymax) / 2.;
    height = ymax - ymin;
    ymin = y_center - 0.5 * height * 1.2
    ymax = y_center + 0.5 * height * 1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


def process_bbox(bbox, img_width, img_height):
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
    aspect_ratio = opt.input_img_shape[1] / opt.input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * 1.25
    bbox[3] = h * 1.25
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    return bbox

# pytorch version
def make_gaussian_heatmap(joint_coord_img):
    x = torch.arange(hm_shape[2])
    y = torch.arange(hm_shape[1])
    yy, xx = torch.meshgrid(y, x)
    xx = xx[None, None, :, :].float().to(device)
    yy = yy[None, None, :, :].float().to(device)
    x = joint_coord_img[:, :, 0, None, None]
    y = joint_coord_img[:, :, 1, None, None]
    heatmap = torch.exp(
        -(((xx - x) / sigma) ** 2) / 2 - (((yy - y) / sigma) ** 2) / 2)
    return heatmap

def get_aug_config(exclude_flip):
    scale_factor = (0.9, 1.1)
    rot_factor = 180
    color_factor = 0.2
    transl_factor = 10
    # scale_factor = (0.5, 1.5)
    # rot_factor = 180
    # color_factor = 0.5
    # transl_factor = 30
    scale = np.random.rand() * (scale_factor[1] - scale_factor[0]) + scale_factor[0]
    rot = (np.random.rand() * 2 - 1) * rot_factor
    transl_x = (np.random.rand() * 2 - 1) * transl_factor
    transl_y = (np.random.rand() * 2 - 1) * transl_factor
    transl = (transl_x, transl_y)
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    if exclude_flip:
        do_flip = False
    else:
        do_flip = random.random() <= 0.5

    return scale, rot, transl, color_scale, do_flip

def augmentation(img, bbox, data_split, exclude_flip=False):
    if data_split == 'train':
        scale, rot, transl, color_scale, do_flip = get_aug_config(exclude_flip)
    else:
        scale, rot, transl, color_scale, do_flip = 1.0, 0.0, (0.0, 0.0), np.array([1, 1, 1]), False
    img, trans, inv_trans, trans_joint, inv_trans_joint \
        = generate_patch_image(img, bbox, scale, rot, transl, do_flip, opt.input_img_shape)
    img = np.clip(img * color_scale[None, None, :], 0, 255)

    return img, trans, inv_trans, rot, do_flip, inv_trans_joint

def generate_patch_image(cvimg, bbox, scale, rot, transl, do_flip, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, transl)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, transl,
                                        inv=True)
    trans_joint = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], 1.0, 0.0, transl)
    inv_trans_joint = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], 1.0, 0.0,
                                        transl, inv=True)

    return img_patch, trans, inv_trans, trans_joint, inv_trans_joint



def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, transl, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment translation
    src_center[0] += transl[0]
    src_center[1] += transl[1]

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

def read_depth_img(depth_filename):
    """Read the depth image in dataset and decode it"""
    #depth_scale = 0.00012498664727900177
    depth_scale = 0.001
    depth_img = cv2.imread(depth_filename)
    dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
    dpt = dpt * depth_scale * 1000
    return dpt

if __name__ == '__main__':
    hand_side = 1

