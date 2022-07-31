# -*- coding: utf-8 -*-
import os
import argparse
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import numpy as np
from PIL import Image
import random
import pickle
import time
import cv2
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from .rotation import rotate_pose_param,rotate_img
from .utils import draw_umich_gaussian

jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]


jointsMapManoToSMPLX = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,20,19,16]
jointsMapSMPLXToMano=[0,1,2,3,16,4,5,6,17,7,8,9,18,10,11,12,19,13,14,15,20]
def project_3D_points(cam_mat, pts3D, is_OpenGL_coords=True):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    # notice: 2020.11.25 landmark翻转.
    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if is_OpenGL_coords:
       pts3D = pts3D.dot(coord_change_mat.T)

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:,0]/proj_pts[:,2], proj_pts[:,1]/proj_pts[:,2]],axis=1)

    assert len(proj_pts.shape) == 2
    return proj_pts

def db_size(set_name):
    """ Hardcoded size of the datasets. """
    if set_name == 'train':
        return 65920  #66034  # number of unique samples (they exists in multiple 'versions')
    elif set_name == 'evaluation':
        return 11524
    else:
        assert 0, 'Invalid choice.'

def get_transform(split_name):
    #normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    normalizer = transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))
    t_list = []
    if split_name == 'training':
        #t_list = [transforms.RandomResizedCrop(224),
        #          transforms.RandomHorizontalFlip()]
        t_list = [#transforms.RandomResizedCrop(224),
                  #transforms.Resize(224),
                  transforms.Resize(224),
                  #transforms.RandomRotation(30),
                  transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                  #transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                  #transforms.RandomHorizontalFlip()
                 ]
    elif split_name == 'val':
#         t_list = [transforms.Resize(224)]  #, transforms.CenterCrop(224)]
        t_list = [transforms.Resize(224)]  #, transforms.CenterCrop(224)]
    elif split_name == 'test':
#         t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
        t_list = [transforms.Resize((224,224))]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform

def rescale_3d_joints(joints_3d):
    x1=joints_3d[4][0]
    y1=joints_3d[4][1]
    z1=joints_3d[4][2]
    x2=joints_3d[5][0]
    y2=joints_3d[5][1]
    z2=joints_3d[5][2]
    l=np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1 - z2)*(z1-z2))
    scale=0.03058954/l
    joints_3d=joints_3d*scale
    joints_3d-=joints_3d[1]
    joints_3d = joints_3d * np.array([1, -1, -1])  #rotate 3D joints 180 at x-axis
    return joints_3d

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

def convert_rotvec_to_quat(pose):
    new_pose = R.from_rotvec(pose).as_quat()[:,[3,0,1,2]]
    quat_array = np.zeros((pose.shape[0],4))
    for i, cur_pose in enumerate(new_pose):
        quat = Quaternion(cur_pose)
        # log_map = Quaternion.log_map(Quaternion(),quat).q[1:]
        quat_array[i,:] = quat.q

    return quat_array

def convert_quat_to_rotvec(quat_pose):
    
    output = np.zeros((quat_pose.shape[0],3))
    for i, quat in enumerate(quat_pose):
        rot = R.from_quat(quat[[1,2,3,0]])
        rotvec = rot.as_rotvec()
        output[i,:] = rotvec

    return output
def rotate_pose(pose):
    quat_array=convert_rotvec_to_quat(pose[None,:3])
    q1=Quaternion(axis=[1,0,0],angle=3.14159265) #rotate 180 at x-axis
    quat_array=q1*quat_array[0]
    quat_array = quat_array.q[None, :]
    pose[:3]=convert_quat_to_rotvec(quat_array)[0]
    
    
    return pose

class TestDataset(Dataset):
    def __init__(self, transform=None,test_dir='experiments/0218',opt=None):
        # image route path
        self.transform = transform
        self.base_path = test_dir
        self.test_list = os.listdir(test_dir)
        
        self.order=opt.order
        self.use_heatmap=opt.use_heatmap
        self.motion_blur=opt.motion_blur
        self.rotation=opt.rotation

    
    def __getitem__(self, index):
        seq=self.test_list[index]
        
        img_rgb_path = os.path.join(self.base_path, seq)
        image_rgb = Image.open(img_rgb_path).convert('RGB')
#         if self.motion_blur:
#             use_blur=random.randint(0, 5)
#             if use_blur==1:
#                 image_rgb=motion_blur(image_rgb)
        image = self.transform(image_rgb)
        
        # image, w, h, name.
        return image, image_rgb.size[0], image_rgb.size[1], seq

    def __len__(self):
        return len(self.test_list)

def get_loader_test(test_dir, opt):
    transform = get_transform('test')
    dataset = TestDataset(transform, test_dir, opt)
    return dataset