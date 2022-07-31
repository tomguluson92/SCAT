# Copyright (c) Liuhao Ge. All Rights Reserved.

"""
    @date:  2022.07.28  week31
    @note:  user should use their own dataset since our data is load by internal tools with credential.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import oss2 as oss
import scipy.io as sio
import os.path as osp
import logging
import cv2
import numpy as np
import numpy.linalg as LA
import math
import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image,ImageOps
import pickle
import random
import cv2
from .rotation import rotate_img
#from .utils import landmark_to_heatmap
from .utils import draw_umich_gaussian

#from hand_shape_pose.util.image_util import crop_pad_im_from_bounding_rect

jointsMapSimpleToSMPLX=[0,5,6,7,9,10,11,17,18,19,13,14,15,1,2,3,8,12,20,16,4]
jointsMapSMPLXToMano=[0,1,2,3,16,4,5,6,17,7,8,9,18,10,11,12,19,13,14,15,20]
BB_base = 120.054 / 10.0  # cm
BB_fx = 822.79041
BB_fy = 822.79041
BB_tx = 318.47345
BB_ty = 250.31296

SK_fx_color = 607.92271
SK_fy_color = 607.88192
SK_tx_color = 314.78337
SK_ty_color = 236.42484

def get_transform(split_name):
    #normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    normalizer = transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))
    t_list = []
    if split_name == 'training':
        t_list = [transforms.Resize(224),
#                   transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
                  #transforms.RandomResizedCrop(224),
                  #transforms.RandomHorizontalFlip()
                  ]
    elif split_name == 'val':
        t_list = [transforms.Resize(224)]
    elif split_name == 'test':
        t_list = [transforms.Resize(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform

def hand_flip(image, joints_2d):
    image = ImageOps.mirror(image)
    joints_2d[:, 0] *= -1
    width,height=image.size
    joints_2d=[width,0]+joints_2d
    return image, joints_2d

def crop_hand(image,joints_2d):
    crop_center=joints_2d[4]
    min_coord = np.maximum(joints_2d.min(0),[0,0])
    max_coord = np.minimum(joints_2d.max(0), np.array(image.size))
    crop_size_best = 1.3 * np.maximum(max_coord - crop_center, crop_center - min_coord)
    crop_size_best = np.max(crop_size_best)
    crop_size_best = min(max(crop_size_best, 10), 500)
    left,top=crop_center-crop_size_best
    right,bottom=crop_center+crop_size_best
    image=image.crop((left,top,right,bottom))

    new_width,new_height=image.size
    image=image.resize((224,224))
    dx=np.array([-left, 0])
    dy=np.array([0, -top])
    scale=224/new_width
    #image = ImageOps.mirror(image)
    joints_2d=(joints_2d+dx+dy)*scale
    #joints_2d[:, 0] *= -1
    #joints_2d=[224,0]+joints_2d
    return image, joints_2d

def rescale_3d_joints_flip(joints_3d):
    x1=joints_3d[4][0]
    y1=joints_3d[4][1]
    z1=joints_3d[4][2]
    x2=joints_3d[5][0]
    y2=joints_3d[5][1]
    z2=joints_3d[5][2]
    l=np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1 - z2)*(z1-z2))
    scale=0.03058954/l
    joints_3d=joints_3d*scale
    joints_3d[:, 0] *= -1
    joints_3d-=joints_3d[1]
    return joints_3d

def SK_rot_mx(rot_vec):
    """
    use Rodrigues' rotation formula to transform the rotation vector into rotation matrix
    :param rot_vec:
    :return:
    """
    theta = LA.norm(rot_vec)
    vector = np.array(rot_vec) * math.sin(theta / 2.0) / theta
    a = math.cos(theta / 2.0)
    b = -vector[0]
    c = -vector[1]
    d = -vector[2]
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c + a * d), 2 * (b * d - a * c)],
                     [2 * (b * c - a * d), a * a + c * c - b * b - d * d, 2 * (c * d + a * b)],
                     [2 * (b * d + a * c), 2 * (c * d - a * b), a * a + d * d - b * b - c * c]])


SK_rot_vec = [0.00531, -0.01196, 0.00301]
SK_trans_vec = [-24.0381, -0.4563, -1.2326]  # mm
SK_rot = SK_rot_mx(SK_rot_vec)

STB_joints = ['loc_bn_palm_L', 'loc_bn_pinky_L_01', 'loc_bn_pinky_L_02', 'loc_bn_pinky_L_03',
              'loc_bn_pinky_L_04', 'loc_bn_ring_L_01', 'loc_bn_ring_L_02', 'loc_bn_ring_L_03',
              'loc_bn_ring_L_04', 'loc_bn_mid_L_01', 'loc_bn_mid_L_02', 'loc_bn_mid_L_03',
              'loc_bn_mid_L_04', 'loc_bn_index_L_01', 'loc_bn_index_L_02', 'loc_bn_index_L_03',
              'loc_bn_index_L_04', 'loc_bn_thumb_L_01', 'loc_bn_thumb_L_02', 'loc_bn_thumb_L_03',
              'loc_bn_thumb_L_04'
              ]
snap_joint_names = ['loc_bn_palm_L', 'loc_bn_thumb_L_01', 'loc_bn_thumb_L_02', 'loc_bn_thumb_L_03',
                    'loc_bn_thumb_L_04', 'loc_bn_index_L_01', 'loc_bn_index_L_02', 'loc_bn_index_L_03',
                    'loc_bn_index_L_04', 'loc_bn_mid_L_01', 'loc_bn_mid_L_02', 'loc_bn_mid_L_03',
                    'loc_bn_mid_L_04', 'loc_bn_ring_L_01', 'loc_bn_ring_L_02', 'loc_bn_ring_L_03',
                    'loc_bn_ring_L_04', 'loc_bn_pinky_L_01', 'loc_bn_pinky_L_02', 'loc_bn_pinky_L_03',
                    'loc_bn_pinky_L_04'
                    ]
snap_joint_name2id = {w: i for i, w in enumerate(snap_joint_names)}
STB_joint_name2id = {w: i for i, w in enumerate(STB_joints)}
STB_to_Snap_id = [snap_joint_name2id[joint_name] for joint_name in STB_joints]
STB_ori_dim = [480, 640]
resize_dim = [256, 256]


def motion_blur(img):
    img=np.asarray(img)
    # Specify the kernel size. 
    # The greater the size, the more the motion. 

    kernel_size = random.randint(1, 10)

    # Create the vertical kernel. 
    kernel_v = np.zeros((kernel_size, kernel_size)) 

    # Create a copy of the same for creating the horizontal kernel. 
    kernel_h = np.copy(kernel_v) 

    # Fill the middle row with ones. 
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 

    # Normalize. 
    kernel_v /= kernel_size 
    kernel_h /= kernel_size 

    # Apply the vertical kernel. 
    vertical_mb = cv2.filter2D(img, -1, kernel_v) 

    # Apply the horizontal kernel. 
    horizonal_mb = cv2.filter2D(img, -1, kernel_h) 
    choice=random.randint(0, 1)
    if choice==0:
        blur_img=vertical_mb
    else:
        blur_img=horizonal_mb
    blur_img = Image.fromarray(np.uint8(blur_img)).convert('RGB')

    return blur_img

class STBDataset(torch.utils.data.Dataset):
    def __init__(self,name="STB_train",transform=None,rotation=False,motion_blur=False,opt=None):
        self.rotation=rotation
        self.motion_blur=motion_blur
        self.use_heatmap=opt.use_heatmap
        DATASETS = {
            "STB_train": {
                "image_list": ["B2Counting", "B2Random","B3Counting", "B3Random","B4Counting", "B4Random","B5Counting", "B5Random","B6Counting", "B6Random",],
                "image_prefix": "SK_color",
            },
            "STB_eval": {
                "image_list": ["B1Counting", "B1Random"],
                "image_prefix": "SK_color",
            }
        }
        attrs = DATASETS[name]
        self.bucket = None  # TODO: user need to specify this to local path
        self.base_path = './STB/'
        self.ann_dir = os.path.join(self.base_path, 'labels')

        self.image_paths = []
        self.bboxes = []
        self.pose_roots = []
        self.pose_scales = []
        self.pose_gts = []
        self.cam_params = torch.tensor([SK_fx_color, SK_fy_color, SK_tx_color, SK_ty_color])
        self.colorKmat = [[607.92271, 0, 314.78337], [0, 607.88192, 236.42484], [0, 0, 1]]
        root_id = snap_joint_name2id['loc_bn_palm_L']

        image_dir_list = [os.path.join(self.base_path, image_dir) for image_dir in attrs["image_list"]]
        ann_file_list = [os.path.join(self.ann_dir, image_dir + "_" + attrs["image_prefix"][:2] + ".pkl")
                         for image_dir in attrs["image_list"]]
        image_prefix=attrs["image_prefix"]
        for image_dir, ann_file in zip(image_dir_list, ann_file_list):
            #the_file=self.bucket.get_object(ann_file)
            #the_file.seek(0, os.SEEK_END)
            #mat_gt = sio.loadmat(the_file)
            try:
                rawdata = self.bucket.get_object(ann_file).read()
                mat_gt = pickle.loads(rawdata, encoding='latin1')
            except:
                mat_gt = self.bucket.get_object(ann_file).read()
                anno=pickle.loads(rawdata)
            curr_pose_gts = mat_gt["handPara"].transpose((2, 1, 0))  # N x K x 3
            curr_pose_gts = self.SK_xyz_depth2color(curr_pose_gts, SK_trans_vec, SK_rot)
            curr_pose_gts = curr_pose_gts[:, STB_to_Snap_id, :] / 1000.0  # convert to Snap index, mm->m
            curr_pose_gts = self.palm2wrist(curr_pose_gts)  # N x K x 3
            curr_pose_gts = torch.from_numpy(curr_pose_gts)
            self.pose_gts.append(curr_pose_gts)

            self.pose_roots.append(curr_pose_gts[:, root_id, :])  # N x 3
            self.pose_scales.append(self.compute_hand_scale(curr_pose_gts))  # N

            for image_id in range(curr_pose_gts.shape[0]):
                self.image_paths.append(osp.join(image_dir, "%s_%d.png" % (image_prefix, image_id)))

        self.pose_roots = torch.cat(self.pose_roots, 0).float()
        self.pose_scales = torch.cat(self.pose_scales, 0).float()
        self.pose_gts = torch.cat(self.pose_gts, 0).float()
        self.transform=transform
        #mat_bboxes = sio.loadmat(bbox_file)
        #self.bboxes = torch.from_numpy(mat_bboxes["bboxes"]).float()  # N x 4

    def __getitem__(self, index):

        image = Image.open(self.bucket.get_object(self.image_paths[index])).convert('RGB')
        joints_3d=self.pose_gts[index][jointsMapSimpleToSMPLX]
        
        kp_coord_uv_proj = np.matmul(joints_3d, np.transpose(self.colorKmat))
        joints_2d = kp_coord_uv_proj[:, :2] / kp_coord_uv_proj[:, 2:]
        joints_2d=joints_2d.cpu().detach().numpy() 
        joints_3d=joints_3d.cpu().detach().numpy()
        joints_3d=rescale_3d_joints_flip(joints_3d)
        
        if self.transform is not None:
            image,joints_2d=hand_flip(image,joints_2d)
            if self.motion_blur:
                use_blur=random.randint(0, 5)
                if use_blur==1:
                    image=motion_blur(image)
            #apply rotation
            if self.rotation:
                angle=random.randint(1, 360)
                image,joints_2d,joints_3d=rotate_img(image,joints_2d,joints_3d,angle)
            image,joints_2d=crop_hand(image,joints_2d)  #This dataset only contains left hand, should be flip to right hand
            image = self.transform(image)
        
        
        if self.use_heatmap:
            heatmap = []
            
            for i in range(len(joints_2d)):
                ht = np.zeros((56, 56))
                ht = draw_umich_gaussian(ht, (joints_2d[i][0]/4, joints_2d[i][1]/4), 8)
                heatmap.append(ht)
            heatmap=torch.tensor(heatmap)
        
        joints_3d=np.array(joints_3d).reshape(-1).tolist()
        joints_2d=np.array(joints_2d).reshape(-1).tolist()
        
        target = np.append(joints_3d, joints_2d)
        #return image, np.array(target), heatmap
        if self.use_heatmap:
            return image,np.array(target),heatmap
        else:
            return image,np.array(target)

    def __len__(self):
        return len(self.image_paths)

    def SK_xyz_depth2color(self, depth_xyz, trans_vec, rot_mx):
        """
        :param depth_xyz: N x 21 x 3, trans_vec: 3, rot_mx: 3 x 3
        :return: color_xyz: N x 21 x 3
        """
        color_xyz = depth_xyz - np.tile(trans_vec, [depth_xyz.shape[0], depth_xyz.shape[1], 1])
        return color_xyz.dot(rot_mx)

    def palm2wrist(self, pose_xyz):
        root_id = snap_joint_name2id['loc_bn_palm_L']
        ring_root_id = snap_joint_name2id['loc_bn_ring_L_01']
        pose_xyz[:, root_id, :] = pose_xyz[:, ring_root_id, :] + \
                                  2.0 * (pose_xyz[:, root_id, :] - pose_xyz[:, ring_root_id, :])  # N x K x 3
        return pose_xyz

    def compute_hand_scale(self, pose_xyz):
        ref_bone_joint_1_id = snap_joint_name2id['loc_bn_mid_L_02']
        ref_bone_joint_2_id = snap_joint_name2id['loc_bn_mid_L_01']

        pose_scale_vec = pose_xyz[:, ref_bone_joint_1_id, :] - pose_xyz[:, ref_bone_joint_2_id, :]  # N x 3
        pose_scale = torch.norm(pose_scale_vec, dim=1)  # N
        return pose_scale

    def evaluate_pose(self, results_pose_cam_xyz, save_results=False, output_dir=""):
        avg_est_error = 0.0
        for image_id, est_pose_cam_xyz in results_pose_cam_xyz.items():
            dist = est_pose_cam_xyz - self.pose_gts[image_id]  # K x 3
            avg_est_error += dist.pow(2).sum(-1).sqrt().mean()

        avg_est_error /= len(results_pose_cam_xyz)

        if save_results:
            eval_results = {}
            image_ids = results_pose_cam_xyz.keys()
            image_ids.sort()
            eval_results["image_ids"] = np.array(image_ids)
            eval_results["gt_pose_xyz"] = [self.pose_gts[image_id].unsqueeze(0) for image_id in image_ids]
            eval_results["est_pose_xyz"] = [results_pose_cam_xyz[image_id].unsqueeze(0) for image_id in image_ids]
            eval_results["gt_pose_xyz"] = torch.cat(eval_results["gt_pose_xyz"], 0).numpy()
            eval_results["est_pose_xyz"] = torch.cat(eval_results["est_pose_xyz"], 0).numpy()
            sio.savemat(osp.join(output_dir, "pose_estimations.mat"), eval_results)

        return avg_est_error.item()
    
    
def get_loader_STB(stage='training', bs=128,opt=None):
    assert stage in ('training', 'val', 'test')
    transform = get_transform(stage)
    rotation=opt.rotation
    motion_blur=opt.motion_blur
    dataset = STBDataset("STB_train",transform,rotation,motion_blur,opt)
    if stage is 'training':
        data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    return dataset

def get_loader_STB_eval(opt):
    stage='val'
    assert stage in ('training', 'val', 'test')
    transform = get_transform(stage)
    opt.use_heatmap=False
    dataset = STBDataset("STB_eval",transform,False,False,opt)
    
    
    return dataset