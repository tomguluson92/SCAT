# -*- coding: utf-8 -*-
"""
    @date:     2021.02.08 星期一 week6
    @readme:   把代码改为我的新transformer的.
    
    @date:     2021.02.19 星期五 week7
    @update:   对比frankmocap, interHand以及Minimal Hand的结果.
    
    @date:     2021.02.23 星期二 week8
    @update:   比较使用了pose length reg loss的模型在Ho3D, MHP, STB这3个时序数据集上的效果.
    
    @date:     2021.02.24 星期三 week8
    @update:   可视化feature map并保存.
    
    @date:     2021.03.09 星期二 week10
    @update:   可视化attention结果.
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.hand_net import EncoderTransformer, EncoderTransformerCoarse
from models.mano import ManoHand,rot_pose_beta_to_mesh
from models.hand_net import H3DWEncoder

import numpy as np
import pickle
import os
import os.path as osp
from torch.nn.parallel import DistributedDataParallel
import cv2
from config import BaseOptions
from dataset.MultiDataset import concat_dataset
from dataset.load_STB import get_loader_STB_eval
from dataset.load_frei_3d import get_loader_frei_eval, get_loader_frei
from dataset.load_ho3d_ding import get_loader_ho3d_eval, get_loader_ho3d
from dataset.load_test_dataset import get_loader_test
import time
import matplotlib.pyplot as plt


import collections
from data_utils.eval_utils import compute_accel,compute_error_accel
from dataset.inference import Inference,MHP_eval, project_3D_points,crop_hand,get_default_transform
from dataset.load_STB import STB_VIBE_demo #load STB
from dataset.load_ho3d_ding import ho3d_VIBE_demo #load ho3d

jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]
jointsMapSMPLXToSimple = [0,
                          13, 14, 15, 20,
                          1, 2, 3, 16,
                          4, 5, 6, 17,
                          10, 11, 12, 19,
                          7, 8, 9, 18]
color_hand_joints = [[1.0, 0.0, 0.0],
                     [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                     [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                     [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                     [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                     [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little


import glob

def generate_video(pth, out_pth):

    img_array = []
    for filename in glob.glob(f'{pth}/*.png'):
        img = cv2.imread(filename)
        #print('Read a frame from: ', filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(f'{out_pth}/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def crop_hand_ref(image,joints_2d,joints_2d_ref):
    crop_center=joints_2d_ref[4] #4? 9?
    min_coord = np.maximum(joints_2d_ref.min(0),[0,0])
    max_coord = np.minimum(joints_2d_ref.max(0), np.array(image.size))
    crop_size_best = 1.5 * np.maximum(max_coord - crop_center, crop_center - min_coord)
    crop_size_best = np.max(crop_size_best)
#     crop_size_best = min(max(crop_size_best, 50), 500)
    crop_size_best = min(max(crop_size_best, 20), 500)
    left,top=crop_center-crop_size_best
    right,bottom=crop_center+crop_size_best

    #image = Image.fromarray(image.astype('uint8'), 'RGB')
    image=image.crop((left,top,right,bottom))

    new_width,new_height=image.size
    image=image.resize((224,224))
    dx=np.array([-left, 0])
    dy=np.array([0, -top])
    scale=224/new_width
    return image,(joints_2d+dx+dy)*scale

def batch_compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0,2,1)

    return S1_hat

def plot_2d_hand(axis, coords_hw, vis=None, color_fixed=None, linewidth='1', order='hw', draw_kp=True):
    """ Plots a hand stick figure into a matplotlib figure. """
    if order == 'uv':
        coords_hw = coords_hw[:, ::-1]
    
    colors = np.array(color_hand_joints)
    # define connections and colors of the bones
    bones = [((0, 1), colors[1, :]),
             ((1, 2), colors[2, :]),
             ((2, 3), colors[3, :]),
             ((3, 4), colors[4, :]),

             ((0, 5), colors[5, :]),
             ((5, 6), colors[6, :]),
             ((6, 7), colors[7, :]),
             ((7, 8), colors[8, :]),

             ((0, 9), colors[9, :]),
             ((9, 10), colors[10, :]),
             ((10, 11), colors[11, :]),
             ((11, 12), colors[12, :]),

             ((0, 13), colors[13, :]),
             ((13, 14), colors[14, :]),
             ((14, 15), colors[15, :]),
             ((15, 16), colors[16, :]),

             ((0, 17), colors[17, :]),
             ((17, 18), colors[18, :]),
             ((18, 19), colors[19, :]),
             ((19, 20), colors[20, :])]

    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0

    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue

        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth)

    if not draw_kp:
        return

    for i in range(21):
        if vis[i] > 0.5:
            axis.plot(coords_hw[i, 1], coords_hw[i, 0], 'o', color=colors[i, :])
            axis.text(coords_hw[i, 1], coords_hw[i, 0], '{}'.format(i), fontsize=5, color='white')
            
def plot_3d_hand(ax,pose_cam_xyz):
    """
    :param pose_cam_xyz: 21 x 3
    :param image_size: H, W
    :return:
    """
    assert pose_cam_xyz.shape[0] == 21

    #fig = plt.figure()
    #fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)

    #ax = plt.subplot(111, projection='3d')
    marker_sz = 15
    line_wd = 2

    for joint_ind in range(pose_cam_xyz.shape[0]):
        ax.plot(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1],
                pose_cam_xyz[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        #ax.text(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1],
        #        pose_cam_xyz[joint_ind:joint_ind + 1, 2], '{}'.format(joint_ind), fontsize=5, color='white')
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(pose_cam_xyz[[0, joint_ind], 0], pose_cam_xyz[[0, joint_ind], 1], pose_cam_xyz[[0, joint_ind], 2],
                    color=color_hand_joints[joint_ind], lineWidth=line_wd)
        else:
            ax.plot(pose_cam_xyz[[joint_ind - 1, joint_ind], 0], pose_cam_xyz[[joint_ind - 1, joint_ind], 1],
                    pose_cam_xyz[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
                    linewidth=line_wd)

    ax.axis('auto')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
#     ax.view_init(elev=-90, azim=-90)
    
def rescale_3d_joints(pred,gt):
    pred=pred.cpu().numpy()
    gt=gt.cpu().numpy()
    for idx in range(pred.shape[0]):
        gtDist = np.linalg.norm(gt[idx, 4, :] -
                                 gt[idx, 5, :])
        predDist = np.linalg.norm(pred[idx, 4, :] -
                                 pred[idx, 5, :])
        #print('gtDist: '+str(gtDist))
        #print('predDist: '+str(predDist))
        scale=gtDist/predDist
        pred[idx]=pred[idx]*scale
        #joints_3d[:, 0] *= -1
        gt[idx]-=gt[idx][1]
        pred[idx]-=pred[idx][1]
        
    pred=torch.from_numpy(pred).float().cuda()
    gt=torch.from_numpy(gt).float().cuda()
    return pred,gt

def _getDistPCK(pred, gt, norm_lm_ids):
    """
    Calculate the pck distance for all given poses.
    norm_lm_ids: Use the distance of these landmarks for normalization. Usually
                 lshoulder and rhip.
    """
    pred=pred.cpu().numpy()
    gt=gt.cpu().numpy()
    #print(pred.shape)
    dist = np.empty((1, pred.shape[1], pred.shape[0]))
    #print(dist.shape)
    for imgidx in range(pred.shape[0]):
        # torso diameter
        refDist = np.linalg.norm(gt[imgidx, norm_lm_ids[0], :] -
                                 gt[imgidx, norm_lm_ids[1], :])
        # distance to gt joints
        #print(refDist)
        dist[0, :, imgidx] =\
            np.sqrt(
                np.sum(
                    np.power(pred[imgidx, :, :] -
                             gt[imgidx, :, :], 2),
                    axis=2)) / refDist
        #print(dist.shape)
    return dist

def cal_PCK(pred_joints, gt_joints,rnge):
    
    dists=torch.sqrt(((pred_joints*1000 - gt_joints*1000) ** 2).sum(dim=-1)).cpu().numpy()
    #print(dists.shape)
    dist=dists #*1000
    """Compute PCK values for given joint distances and a range."""
    
    pck = np.zeros((len(rnge), dist.shape[1] + 1))
    for joint_idx in range(dist.shape[1]):
        # compute PCK for each threshold
        for k, rngval in enumerate(rnge):
            pck[k, joint_idx] = 100. *\
                np.mean(dist.flat <= rngval)
    # compute average PCK
    for k in range(len(rnge)):
        pck[k, -1] = np.mean(pck[k, :-1])
    return pck

    #dists = np.sqrt(np.sum((pred_joints - gt_joints) ** 2, axis=2))
    #pck_thresh=40
    #correct = dists <= pck_thresh
    
    #total=dists.shape[0]*dists.shape[1]
    
    #pck=(correct).sum()/total
    #print(pck)
    #print(np.mean(dists.flat <= pck_thresh))
    #return pck
def _area_under_curve(xpts, ypts):
    """Calculate the AUC."""
    a = np.min(xpts)
    b = np.max(xpts)
    
    # remove duplicate points
    _, I = np.unique(xpts, return_index=True)  # pylint: disable=W0632
    
    xpts = xpts[I]
    ypts = ypts[I]
    norm_factor = np.trapz(np.ones_like(xpts), xpts)
    auc=(np.trapz(ypts,xpts)/norm_factor)
    return auc

    #assert np.all(np.diff(xpts) > 0)
    #if len(xpts) < 2:
    #    return np.NAN
    #from scipy import integrate
    #myfun = lambda x: np.interp(x, xpts, ypts)
    #auc = integrate.quad(myfun, a, b)[0]
    #return auc

def load_pkl(pkl_file, res_list=None):
    assert pkl_file.endswith(".pkl")
    with open(pkl_file, 'rb') as in_f:
        try:
            data = pickle.load(in_f)
        except UnicodeDecodeError:
            in_f.seek(0)
            data = pickle.load(in_f, encoding='latin1')
    return data


class Trainer():
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.lr = opt.lr
        self.resume = opt.resume
        self.epoches = opt.epoch
        self.model_choice=opt.hand_choice
        self.top_finger_joints_type = 'ave'
        
        self.mean_mano_params = opt.mean_mano_param
        self.load_frank_params()
        
        self.result_dir = opt.result_dir
        if not os.path.exists(opt.result_dir):
            os.makedirs(opt.result_dir, exist_ok=True)
        #self.train_loader = concat_dataset(self.batch_size)
        
        if opt.net == 'reg_transformer':
            print("fine")
            self.load_mano_mean(opt.outside)
            self.net = EncoderTransformer(opt, self.mean_params).cuda()
        elif opt.net == 'reg_transformer_coarse':
            print("coarse")
            self.load_mano_mean(opt.outside)
            self.net = EncoderTransformerCoarse(opt, self.mean_params).cuda()
        elif opt.net == 'frankmocap':
            print("Frankmocap")
            self.mean_mano_params = opt.mean_mano_param
            self.load_frank_params()
            self.net = H3DWEncoder(opt, self.mean_params).cuda()

        self.checkpoint_path = opt.checkpoint_path_eval
        checkpoint_path = self.checkpoint_path
        if not osp.exists(checkpoint_path):
            print(f"Error: {checkpoint_path} does not exists")
            self.success_load = False
        else:
            saved_weights = torch.load(checkpoint_path)
            self.net.load_state_dict(saved_weights)
            print('Checkpoint loaded from: ' + checkpoint_path)
            self.success_load = True
    
    def load_frank_params(self):
        # load mean params first
        mean_param_file = self.mean_mano_params
        mean_vals = load_pkl(mean_param_file)
        total_params_dim = 61
        mean_params = np.zeros((1, total_params_dim))

        # set camera model first
        mean_params[0, 0] = 5.0

        # set pose (might be problematic)
        mean_pose = mean_vals['mean_pose'][3:]
        # set hand global rotation
        mean_pose = np.concatenate((np.zeros((3,)), mean_pose))
        mean_pose = mean_pose[None, :]

        # set shape
        mean_shape = np.zeros((1, 10))
        mean_params[0, 3:] = np.hstack((mean_pose, mean_shape))
        # concat them together
        self.mean_params = torch.from_numpy(mean_params).float()
        self.mean_params.requires_grad = False
        # define global rotation
        self.global_orient = torch.zeros((self.batch_size, 3), dtype=torch.float32).cuda()
        # self.global_orient[:, 0] = np.pi
        self.global_orient.requires_grad = False
    
    def load_mano_mean(self, outside=True):
        dd = pickle.load(open('extra_data/MANO_RIGHT.pkl', 'rb'),encoding='latin1')
        kintree_table = dd['kintree_table']
        id_to_col = {kintree_table[1, i]: i for i in range(kintree_table.shape[1])}
        parent = {i: id_to_col[kintree_table[0, i]] for i in range(1, kintree_table.shape[1])}

        self.faces = dd['f']
        self.verts = dd['v_template']  # zero mean
        
        """
             20     16    17    19   18
               15    3    6    12   9
                 14   2   5   11   8
                   13  1  4  10   7
                         
                        0
        """
        if outside:
            # 手外部(local), 我用blender看的hand_mean.
            self.local_tree = [188, 142, 87, 290, 216, 316, 402, 200, 585, 630, 285, 473, 513, 88, 249, 702, 329, 439, 668, 550, 740]
        else:
            # 手内部(local), 我用blender看的hand_mean.
            self.local_tree = [35, 168, 47, 337, 283, 353, 449, 591, 599, 637, 139, 467, 560, 5, 121, 707, 329, 439, 668, 550, 740]
        
        # blender的索引是从1开始的.
        self.local_tree = list(map(lambda x: x-1, self.local_tree))
        
        # 加回去.
        mean_params = np.zeros((1, 63+3))
        # set camera model first,
        mean_params[0, 0] = 5.0
        # then set standard joint xyz.
        mean_params[0, 3:] = np.hstack(([self.verts[_, :] for _ in self.local_tree]))
        self.mean_params = torch.from_numpy(mean_params).float()
        self.mean_params.requires_grad = False

    def batch_orth_proj_idrot(self, X, camera):
        # camera is (batchSize, 1, 3)
        camera = camera.view(-1, 1, 3)
        X_trans = X[:, :, :2] + camera[:, :, 1:]
        res = camera[:, :, 0] * X_trans.view(X_trans.size(0), -1)
        return res.view(X_trans.size(0), X_trans.size(1), -1)

    def project_2d(self, joints_2d):
        return joints_2d * 112 + 112
    
    def test(self, test_dir='experiments/0219/img'):
        self.result_folder = self.result_dir
        if not os.path.exists(os.path.join(self.result_dir, "img")):
            os.makedirs(os.path.join(self.result_dir, "img"), exist_ok=True)
        
        dataset=get_loader_test(test_dir=test_dir, opt=self.opt)
        self.test_loader = DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1,drop_last=True)
        self.net.eval()
        n=0
        for i, data in enumerate(self.test_loader):
            n+=1
            with torch.no_grad():
                eval_time = time.time()
                inputs, w, h, names = data
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                inputs = inputs.to(device).float()
                
                # Frankmocap version.
#                 feat, outputs = self.net(inputs)
#                 #print(outputs[0])
#                 # get seperate outputs
#                 cam_dim = 3
#                 pose_dim = 48
#                 pred_cam_params = outputs[:, :cam_dim]
#                 pred_pose_params = outputs[:, cam_dim: (cam_dim + pose_dim)]
#                 pred_shape_params = outputs[:, (cam_dim + pose_dim):]

#                 #  get predicted smpl verts and joints,
#                 if self.model_choice=='mano':
#                     # MANO =======================>
#                     rot = pred_pose_params[:,:3]    
#                     theta = pred_pose_params[:,3:]
#                     beta = pred_shape_params
#                     x3d = rot_pose_beta_to_mesh(rot,theta,beta)
#                     pred_joints_3d=x3d[:,:21]

# #                 generate predicted joints 2d
#                 pred_joints_2d = self.batch_orth_proj_idrot(pred_joints_3d, pred_cam_params)
#                 pred_joints_2d = self.project_2d(pred_joints_2d)


                # 2021.02.11 week6 visualize joint'sfeature map.
                outputs, feat_visual = self.net(inputs)
            
                feature_visual = feat_visual[0].cpu().detach().numpy()
                fm_visual = np.clip(feature_visual*127.5+127.5,0,255).astype(np.uint8)
#                 save_pic = [255 - cv2.resize(_, (224,224)) for _ in fm_visual]  # no pl
                save_pic = [cv2.resize(_, (224,224)) for _ in fm_visual]# pl
                blank_pic = (np.ones((224, 224))*255.).astype(np.uint8)

                fm_final_visual = []
                for _ in range(21):
                    fm_final_visual.append(save_pic[_])
                    if _ != 20:
                        fm_final_visual.append(blank_pic)

                im_h = cv2.hconcat(fm_final_visual)

                cv2.imwrite('experiments/0219/fm/'+f'{n:03d}.png',im_h)
                
                
#                 #get seperate outputs
#                 cam_dim=3
#                 pred_cam_params = outputs[:, :cam_dim]
#                 pred_joints_3d = outputs[:, 3:66].view(-1,21,3)

#                 # generate predicted joints 2d
#                 pred_joints_2d = self.batch_orth_proj_idrot(pred_joints_3d, pred_cam_params)
#                 pred_joints_2d = self.project_2d(pred_joints_2d)
                
# #                 import pdb
# #                 pdb.set_trace()
                
#                 info = {}
#                 info["img"] = names[0]
#                 info["width"] = int(w)
#                 info["height"] = int(h)
#                 info["rescale"] = pred_cam_params[0, 0].cpu().numpy().tolist()
#                 info["trans"] = pred_cam_params[0, 1:].cpu().numpy().tolist()
#                 info["3d"] = pred_joints_3d[0, :].cpu().numpy().tolist()
                
#                 names = names[0].split(".")[0]
# #                 with open("{0}/{1}.json".format(test_dir,names),'w') as file_obj:
# #                     json.dump(info, file_obj)
                
#                 pred_3d=pred_joints_3d[0].cpu().detach().numpy()
#                 pred_2d=pred_joints_2d[0].cpu().detach().numpy()
#                 fig = plt.figure()
#                 fig.set_size_inches(float(1500) / fig.dpi, float(500) / fig.dpi, forward=True)
#                 ax1 = fig.add_subplot(131,projection='3d')
#                 ax2 = fig.add_subplot(132)
#                 image=inputs[0].permute(1, 2, 0).cpu().detach().numpy()
#                 image=np.clip(image*127.5+127.5,0,255).astype(np.uint8)
#                 ax2.imshow(image)
#                 plot_3d_hand(ax1,pred_3d[jointsMapSMPLXToSimple])
#                 ax1.set_xlabel('predict 3d joints', fontsize=10)
#                 plot_2d_hand(ax2,pred_2d[jointsMapSMPLXToSimple],order='uv')
#                 jointsMapManoToSimple=[0,5,6,7,8,9,10,11,12,17,18,19,20,13,14,15,16,1,2,3,4]
#                 #plot_3d_hand(ax1,gt_3d[jointsMapManoToSimple])
#                 #plot_3d_hand(ax2,pred_3d[jointsMapManoToSimple])
#                 fig.savefig(self.result_folder+f'gt_pred_{n:03d}.png')
#                 open_cv_image = image#*255
#                 #open_cv_image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
#                 # Convert RGB to BGR 
#                 open_cv_image = open_cv_image[:, :, ::-1].copy()
#                 cv2.imwrite(self.result_folder+f'img/{n:03d}.png',open_cv_image)
#                 #plt.show()
#                 plt.close()
    
    @torch.no_grad()
    def demo(self, eval_set='ho3d'):
        result_folder=self.result_dir
        if not os.path.exists(os.path.join(result_folder, 'fm')):
            os.makedirs(os.path.join(result_folder, 'fm'))
        if not os.path.exists(os.path.join(result_folder, '3d')):
            os.makedirs(os.path.join(result_folder, '3d'))
        if not os.path.exists(os.path.join(result_folder, 'img')):
            os.makedirs(os.path.join(result_folder, 'img'))
            
        
        self.batch_size = 1
        accelerate_avg = 0
        
        if eval_set=='MHP':
            print("MHP")
#             seq_name='data_1_cam_1'
            seq_name='data_15_cam_1'
            loader=MHP_eval(seq_name)
        elif eval_set=='STB':
            print("STB")
            seq_name='B1Counting'
            #seq_name='B1Random'
            loader = STB_VIBE_demo(seq_name)
        elif eval_set=='ho3d':
            print("ho3d")
            seq_name='GPMF11'
            loader = ho3d_VIBE_demo(seq_name)
        
        time_seq = min(loader.seq_len(), 200)
        mpjpe = np.zeros(time_seq)
        print(f'Input {seq_name} video number of frames {loader.seq_len()}')
        
        cur_frame=0
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.net.eval()

        auc_3d=[]
        all_pred=None
        rnge = np.arange(20, 51,5)
        pck_all = np.zeros((len(rnge), 21 + 1))
        n=0
        
        acc_list, tar_list = [], []

        for i in range(time_seq):
            eval_time = time.time()
            img, kp_2d, kp_3d = loader.get_sample(i)
            n+=1
            if cur_frame==0:
                kp_2d_ref=kp_2d
            fix_window=True
#             fix_window=False
            if fix_window:
                img,kp_2d=crop_hand_ref(img,kp_2d,kp_2d_ref)
            else:
                img,kp_2d=crop_hand(img,kp_2d)

            transform = get_default_transform()
            image = transform(img)
            image = image.unsqueeze(0).to(device).float()

            outputs, feature_visual = self.net(image)  # 不用pose length reg的. 2021.02.23 week8 星期二
            
            feature_visual = feature_visual[0].cpu().detach().numpy()
            fm_visual = np.clip(feature_visual*127.5+127.5,0,255).astype(np.uint8)
            save_pic = [255 - cv2.resize(_, (224,224)) for _ in fm_visual]  # no pl
#             save_pic = [cv2.resize(_, (224,224)) for _ in fm_visual]# pl
            blank_pic = (np.ones((224, 224))*255.).astype(np.uint8)
            
            fm_final_visual = []
            for _ in range(21):
                fm_final_visual.append(save_pic[_])
                if _ != 20:
                    fm_final_visual.append(blank_pic)
                    
            im_h = cv2.hconcat(fm_final_visual)
            
            cv2.imwrite(result_folder+f'fm/{n:03d}.png', im_h)
            
            cam_dim=3
            pred_cam_params = outputs[:, :cam_dim]
            pred_joints_3d = outputs[:, 3:66].view(-1,21,3).detach()

            # generate predicted joints 2d
            pred_joints_2d = self.batch_orth_proj_idrot(pred_joints_3d, pred_cam_params)
            pred_joints_2d = self.project_2d(pred_joints_2d)
            cur_frame+=1
            
            gt_joints_3d=torch.from_numpy(kp_3d).view(21,3).cuda().unsqueeze(0)
            gt_joints_2d=torch.from_numpy(kp_2d).view(21,2).cuda().unsqueeze(0)
            
            if len(acc_list) < 16:
                acc_list.append(pred_joints_3d[0].cpu().detach().numpy())
                tar_list.append(gt_joints_3d[0].cpu().numpy())
            else:
                acc_list.pop(0)
                tar_list.pop(0)
                acc_list.append(pred_joints_3d[0].cpu().detach().numpy())
                tar_list.append(gt_joints_3d[0].cpu().numpy())
            
            # 3) calculate acceleration
            m2mm=1000
            if len(acc_list) == 16:
                accel = np.mean(compute_accel(np.asarray(acc_list))) * m2mm
                print('acceleration: ' + str(accel))
                accelerate_avg += accel
                accel_err = np.mean(compute_error_accel(joints_pred=np.asarray(acc_list), joints_gt=np.asarray(tar_list))) * m2mm
                print('acceleration error (compare with gt): ' + str(accel_err))

            end = time.time()
            fps = self.batch_size / (end - eval_time)
            print(f'FPS: {fps:.2f}')
            #cal 3d pck
            norm_lm_ids=[4,5]
            #dist = _getDistPCK(pred_joints_3d, gt_joints_3d, norm_lm_ids)
            #print(dist)

            rnge = np.arange(20,51,5) #from 20mm to 50mm
#             import pdb
#             pdb.set_trace()
            pck = cal_PCK(pred_joints_3d.detach(), gt_joints_3d, rnge)
            plot_3d=True
            if plot_3d:
                gt_3d=gt_joints_3d[0].cpu().detach().numpy()
                gt_2d=gt_joints_2d[0].cpu().detach().numpy()
                pred_3d=pred_joints_3d[0].cpu().detach().numpy()
                pred_2d=pred_joints_2d[0].cpu().detach().numpy()
                fig = plt.figure()
                fig.set_size_inches(float(1500) / fig.dpi, float(500) / fig.dpi, forward=True)
                ax1 = fig.add_subplot(131,projection='3d')
                ax2 = fig.add_subplot(132,projection='3d')
                ax3 = fig.add_subplot(133)
                image_save = np.asarray(img).astype(np.uint8)

                ax3.imshow(image_save)
                plot_3d_hand(ax1,gt_3d[jointsMapSMPLXToSimple])
                ax1.set_xlabel('ground truth 3d joints', fontsize=10)
                plot_3d_hand(ax2,pred_3d[jointsMapSMPLXToSimple])
                ax2.set_xlabel('predict 3d joints', fontsize=10)
                plot_2d_hand(ax3,pred_2d[jointsMapSMPLXToSimple],order='uv')
#                 plot_2d_hand(ax3,gt_2d[jointsMapSMPLXToSimple],order='uv')
                jointsMapManoToSimple=[0,5,6,7,8,9,10,11,12,17,18,19,20,13,14,15,16,1,2,3,4]
                #plot_3d_hand(ax1,gt_3d[jointsMapManoToSimple])
                #plot_3d_hand(ax2,pred_3d[jointsMapManoToSimple])
                fig.savefig(result_folder+f'3d/gt_pred_{n:03d}.png')

                open_cv_image = np.asarray(image_save)#*255
                #open_cv_image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                # Convert RGB to BGR 
                open_cv_image = open_cv_image[:, :, ::-1].copy()
                if not os.path.exists(os.path.join(result_folder, 'img')):
                    os.makedirs(os.path.join(result_folder, 'img'))
                cv2.imwrite(result_folder+f'img/{n:03d}.png', open_cv_image)
                #plt.show()
                plt.close()
            pck_all+=pck
            auc = _area_under_curve(rnge / rnge.max(), pck[:, -1])

            auc_3d.append(auc)
            print("AUC: {}.".format(auc))

            print("@50: {}.".format(pck[np.argmax(rnge > 50) - 1, -1]))
            #pck_2d = cal_PCK(pred_joints_2d, gt_joints_2d)

            # Absolute error (MPJPE)
            error = torch.sqrt(((pred_joints_3d - gt_joints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            mpjpe[i * self.batch_size:i * self.batch_size + self.batch_size] = error
            
        # 2021.06.15 导出视频.
        generate_video(os.path.join(result_folder, "3d"), result_folder)
        
        
        pck_all=pck_all/n
        auc = _area_under_curve(rnge / rnge.max(), pck_all[:, -1])
        plot=True
        if plot:
            plt.figure(figsize=(7, 7))
            plt.plot(rnge,
                    pck_all[:, -1],
                    label='PCK',
                    linewidth=2)
            plt.xlim(20, 50)
            plt.xticks(np.arange(20, 51, 5))
            plt.yticks(np.arange(0, 101., 10.))
            plt.ylabel('Detection rate, %')
            plt.xlabel('Error Thresholds (mm)')
            plt.grid()
            legend = plt.legend(loc=4)
            legend.get_frame().set_facecolor('white')
            plt.savefig(result_folder+'PCK.png')
            #plt.show()
            plt.close()
            print('*** Final Results ***')
            print()
            print('MPJPE: ' + str(1000 * mpjpe.mean()))
            print('ACC:' + str(accelerate_avg / time_seq))
            #print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
            #print('AUC mean: ' + str( np.array(auc_3d).mean()))
            print('AUC: ' + str(auc))
    
    def eval(self,eval_dataset='STB'):
        result_folder=self.result_dir
        if eval_dataset=='STB':
            dataset=get_loader_STB_eval(self.opt)
        elif eval_dataset=='frei':
            dataset=get_loader_frei(stage='training',bs=self.batch_size,opt=self.opt)
        elif eval_dataset=='ho3d':
            dataset=get_loader_ho3d(stage='training', bs=self.batch_size, opt=self.opt)
        self.eval_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,num_workers=10,drop_last=True)

        self.net.eval()
        #Pose metrics
        mpjpe = np.zeros(len(dataset))
        recon_err = np.zeros(len(dataset))


        auc_3d=[]
        all_pred=None
        rnge = np.arange(20, 51,5)
        pck_all = np.zeros((len(rnge), 21 + 1))
        n=0
        
        for i, data in enumerate(self.eval_loader):
            n+=1
            with torch.no_grad():
                eval_time = time.time()
                inputs, labels = data
                
                # 跳过空数据. 2021.02.08 星期一 week6
                content = abs(inputs.sum(dim=[1,2,3]))
                idxs = torch.where(abs(content - 224*224*3) > 2000, torch.ones_like(content), torch.zeros_like(content))
                idx_lst = idxs * torch.arange(self.batch_size)
                idx_lst = list(map(lambda x: int(x), idx_lst))
                idx_lst = [_ for idx, _ in enumerate(idx_lst) if _ == idx]

                inputs = inputs[idx_lst, ...]
                labels = labels[idx_lst, ...]

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
                
                # 2021.02.11 week6 星期四 可视化不同joint的feature map.
#                 outputs, feat_visual = self.net(inputs)
                
#                 # 2021.02.20 week7 星期六 attn
                outputs, feat_visual, attn = self.net(inputs)
            
                #get seperate outputs
                cam_dim=3
                pred_cam_params = outputs[:, :cam_dim]
                pred_joints_3d = outputs[:, 3:66].view(-1,21,3)

                # generate predicted joints 2d
                pred_joints_2d = self.batch_orth_proj_idrot(pred_joints_3d, pred_cam_params)
                pred_joints_2d = self.project_2d(pred_joints_2d)
                
                #pred_joints_3d = pred_joints_3d.view(-1, 63)
                #pred_joints_2d = pred_joints_2d.view(-1, 42)

                # seperate loss
                if labels.size()[1] == 105:  # MTC, RHD, STB only have 3d & 2d joints gt
                    gt_joints_3d = labels[:, :63]
                    gt_joints_2d = labels[:, 63:]
                else:  # freihand, ho3d have pose, 3d & 2d joints gt
                    gt_pose = labels[:, 3:51]
                    gt_joints_3d = labels[:, 61:(61 + 63)]
                    gt_joints_2d = labels[:, (61 + 63):]
                    
                gt_lmk = gt_joints_2d[1].cpu().view(21,2).numpy()
#                 import pdb
#                 pdb.set_trace()
#                 head_num = attn.size()[1]
#                 attn = attn.sum(dim=1, keepdim=True)
#                 attn = attn / head_num
                
                attn_1 = attn[1, 0, 1].cpu().numpy(); start_1 = gt_lmk[1]    # index
                attn_2 = attn[1, 0, 20].cpu().numpy(); start_2 = gt_lmk[20]  # thumb
                attn_3 = attn[1, 0, 5].cpu().numpy(); start_3 = gt_lmk[5]    # middle
                attn_4 = attn[1, 0, 10].cpu().numpy(); start_4 = gt_lmk[10]  # ring
                attn_5 = attn[1, 0, 18].cpu().numpy(); start_5 = gt_lmk[18]  # little
                
                # 1) 食指
                img_out = np.zeros((224*6,224*6,3),np.uint8)
                attn_sort = np.sort(attn_1)
                for idx, item in enumerate(gt_lmk):
                    if idx != 1:
                        cv2.circle(img_out, (int(item[0]*6), int(item[1]*6)), 5, [255, 255, 255], -1)
                    if idx == 1:
                        cv2.circle(img_out, (int(item[0]*6), int(item[1]*6)), 20, [220, 20, 60], -1)
                    if idx != 1 and attn_1[idx]-attn_sort[5] > 0:
                        print(int((max(attn_1[idx]-attn_sort[5], 0)/ (attn_sort[-1]-attn_sort[5]))*10))
                        if int((max(attn_1[idx]-attn_sort[5], 0)/ (attn_sort[-1]-attn_sort[5]))*10) > 0:
                            cv2.line(img_out, (int(start_1[0]*6), int(start_1[1]*6)), (int(item[0]*6), int(item[1]*6)), (0,255,0), int((max(attn_1[idx]-attn_sort[5], 0)/ (attn_sort[-1]-attn_sort[5]))*10), lineType=4)
                
                cv2.imwrite(result_folder+f'attn/index/{n:03d}.png', img_out)
                
                # 2) 拇指
                img_out = np.zeros((224*6,224*6,3),np.uint8)
                attn_sort = np.sort(attn_2)
                for idx, item in enumerate(gt_lmk):
                    if idx != 20:
                        cv2.circle(img_out, (int(item[0]*6), int(item[1]*6)), 5, [255, 255, 255], -1)
                    if idx == 20:
                        cv2.circle(img_out, (int(item[0]*6), int(item[1]*6)), 20, [220, 20, 60], -1)
                    if idx != 1 and attn_2[idx]-attn_sort[5] > 0:
                        print(int((max(attn_2[idx]-attn_sort[5], 0)/ (attn_sort[-1]-attn_sort[5]))*10))
                        if int((max(attn_2[idx]-attn_sort[5], 0)/ (attn_sort[-1]-attn_sort[5]))*10) > 0:
                            cv2.line(img_out, (int(start_2[0]*6), int(start_2[1]*6)), (int(item[0]*6), int(item[1]*6)), (189,183,107), int((max(attn_2[idx]-attn_sort[5], 0)/ (attn_sort[-1]-attn_sort[5]))*10), lineType=4)
                
                cv2.imwrite(result_folder+f'attn/thumb/{n:03d}.png', img_out)
                
                # 3) 中指
                img_out = np.zeros((224*6,224*6,3),np.uint8)
                attn_sort = np.sort(attn_3)
                for idx, item in enumerate(gt_lmk):
                    if idx != 5:
                        cv2.circle(img_out, (int(item[0]*6), int(item[1]*6)), 5, [255, 255, 255], -1)
                    if idx == 5:
                        cv2.circle(img_out, (int(item[0]*6), int(item[1]*6)), 20, [220, 20, 60], -1)
                    if idx != 1 and attn_3[idx]-attn_sort[5] > 0:
                        print(int((max(attn_3[idx]-attn_sort[5], 0)/ (attn_sort[-1]-attn_sort[5]))*10))
                        if int((max(attn_3[idx]-attn_sort[5], 0)/ (attn_sort[-1]-attn_sort[5]))*10) > 0:
                            cv2.line(img_out, (int(start_3[0]*6), int(start_3[1]*6)), (int(item[0]*6), int(item[1]*6)), (218,112,214), int((max(attn_3[idx]-attn_sort[5], 0)/ (attn_sort[-1]-attn_sort[5]))*10), lineType=4)
                
                cv2.imwrite(result_folder+f'attn/middle/{n:03d}.png', img_out)
                
                
                # 4) 无名指
                img_out = np.zeros((224*6,224*6,3),np.uint8)
                attn_sort = np.sort(attn_4)
                for idx, item in enumerate(gt_lmk):
                    if idx != 10:
                        cv2.circle(img_out, (int(item[0]*6), int(item[1]*6)), 5, [255, 255, 255], -1)
                    if idx == 10:
                        cv2.circle(img_out, (int(item[0]*6), int(item[1]*6)), 20, [220, 20, 60], -1)
                    if idx != 1 and attn_4[idx]-attn_sort[5] > 0:
                        print(int((max(attn_4[idx]-attn_sort[5], 0)/ (attn_sort[-1]-attn_sort[5]))*10))
                        if int((max(attn_4[idx]-attn_sort[5], 0)/ (attn_sort[-1]-attn_sort[5]))*10) > 0:
                            cv2.line(img_out, (int(start_4[0]*6), int(start_4[1]*6)), (int(item[0]*6), int(item[1]*6)), (0,0,205), int((max(attn_4[idx]-attn_sort[5], 0)/ (attn_sort[-1]-attn_sort[5]))*10), lineType=4)
                
                cv2.imwrite(result_folder+f'attn/ring/{n:03d}.png', img_out)
                
                # 5) 小指
                img_out = np.zeros((224*6,224*6,3),np.uint8)
                attn_sort = np.sort(attn_5)
                for idx, item in enumerate(gt_lmk):
                    if idx != 18:
                        cv2.circle(img_out, (int(item[0]*6), int(item[1]*6)), 5, [255, 255, 255], -1)
                    if idx == 18:
                        cv2.circle(img_out, (int(item[0]*6), int(item[1]*6)), 20, [220, 20, 60], -1)
                    if idx != 1 and attn_5[idx]-attn_sort[5] > 0:
                        print(int((max(attn_5[idx]-attn_sort[5], 0)/ (attn_sort[-1]-attn_sort[5]))*10))
                        if int((max(attn_5[idx]-attn_sort[5], 0)/ (attn_sort[-1]-attn_sort[5]))*10) > 0:
                            cv2.line(img_out, (int(start_5[0]*6), int(start_5[1]*6)), (int(item[0]*6), int(item[1]*6)), (135,206,235), int((max(attn_5[idx]-attn_sort[5], 0)/ (attn_sort[-1]-attn_sort[5]))*10), lineType=4)
                
                cv2.imwrite(result_folder+f'attn/little/{n:03d}.png', img_out)
                
                
                gt_joints_3d=gt_joints_3d.view(-1,21,3)
                gt_joints_2d=gt_joints_2d.view(-1,21,2)
                #pred_joints_3d,gt_joints_3d=rescale_3d_joints(pred_joints_3d,gt_joints_3d) #normalize pred to gt length
                #gt_joints_3d,pred_joints_3d=rescale_3d_joints(gt_joints_3d,pred_joints_3d)   #normalize gt to pred length
                
                # PA
                pred_joints_3d=batch_compute_similarity_transform_torch(pred_joints_3d, gt_joints_3d)
                
                end = time.time()
                fps = self.batch_size / (end - eval_time)
                print(f'FPS: {fps:.2f}')
                #cal 3d pck
                norm_lm_ids=[4,5]
                #dist = _getDistPCK(pred_joints_3d, gt_joints_3d, norm_lm_ids)
                #print(dist)
                
                rnge = np.arange(20, 51,5) #from 20mm to 50mm
                pck = cal_PCK(pred_joints_3d, gt_joints_3d,rnge)
                plot_3d=True
                if plot_3d:
                    gt_3d=gt_joints_3d[0].cpu().detach().numpy()
                    pred_3d=pred_joints_3d[0].cpu().detach().numpy()
                    pred_2d=pred_joints_2d[0].cpu().detach().numpy()
                    fig = plt.figure()
                    fig.set_size_inches(float(1500) / fig.dpi, float(500) / fig.dpi, forward=True)
                    ax1 = fig.add_subplot(131,projection='3d')
                    ax2 = fig.add_subplot(132,projection='3d')
                    ax3 = fig.add_subplot(133)
                    image=inputs[0].permute(1, 2, 0).cpu().detach().numpy()
                    #image=image.permute(1,2,0).cpu().numpy()
                    image=np.clip(image*127.5+127.5,0,255).astype(np.uint8)
                    image_save = image.copy()
                    
                    ax3.imshow(image)
                    plot_3d_hand(ax1,gt_3d[jointsMapSMPLXToSimple])
                    ax1.set_xlabel('ground truth 3d joints', fontsize=10)
                    plot_3d_hand(ax2,pred_3d[jointsMapSMPLXToSimple])
                    ax2.set_xlabel('predict 3d joints', fontsize=10)
                    plot_2d_hand(ax3,pred_2d[jointsMapSMPLXToSimple],order='uv')
                    jointsMapManoToSimple=[0,5,6,7,8,9,10,11,12,17,18,19,20,13,14,15,16,1,2,3,4]
                    #plot_3d_hand(ax1,gt_3d[jointsMapManoToSimple])
                    #plot_3d_hand(ax2,pred_3d[jointsMapManoToSimple])
                    fig.savefig(result_folder+f'gt_pred_{n:03d}.png')
                    
                    open_cv_image = image_save#*255
                    #open_cv_image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                    # Convert RGB to BGR 
                    open_cv_image = open_cv_image[:, :, ::-1].copy() 
                    cv2.imwrite(result_folder+f'img/{n:03d}.png', open_cv_image)
                    #plt.show()
                    plt.close()
                pck_all+=pck
                auc = _area_under_curve(rnge / rnge.max(), pck[:, -1])
                
                auc_3d.append(auc)
                print("AUC: {}.".format(auc))
                
                print("@50: {}.".format(pck[np.argmax(rnge > 50) - 1, -1]))
                #pck_2d = cal_PCK(pred_joints_2d, gt_joints_2d)
                plot=False
                if plot:
                    plt.figure(figsize=(7, 7))
                    plt.plot(rnge,
                             pck[:, -1],
                             label='PCK',
                             linewidth=2)
                    plt.xlim(20, 50)
                    plt.xticks(np.arange(20, 51, 5))
                    plt.yticks(np.arange(0, 101., 10.))
                    plt.ylabel('Detection rate, %')
                    plt.xlabel('Error Thresholds (mm)')
                    plt.grid()
                    legend = plt.legend(loc=4)
                    legend.get_frame().set_facecolor('white')
                    plt.savefig(result_folder+f'PCK{n:03d}.png')
                    #plt.show()
                    plt.close()

                # Absolute error (MPJPE)
                error = torch.sqrt(((pred_joints_3d - gt_joints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                mpjpe[i * self.batch_size:i * self.batch_size + self.batch_size] = error
        pck_all=pck_all/n
        auc = _area_under_curve(rnge / rnge.max(), pck_all[:, -1])
        plot=True
        if plot:
            plt.figure(figsize=(7, 7))
            plt.plot(rnge,
                    pck_all[:, -1],
                    label='PCK',
                    linewidth=2)
            plt.xlim(20, 50)
            plt.xticks(np.arange(20, 51, 5))
            plt.yticks(np.arange(0, 101., 10.))
            plt.ylabel('Detection rate, %')
            plt.xlabel('Error Thresholds (mm)')
            plt.grid()
            legend = plt.legend(loc=4)
            legend.get_frame().set_facecolor('white')
            plt.savefig(result_folder+'PCK.png')
            #plt.show()
            plt.close()
        print('*** Final Results ***')
        print()
        print('MPJPE: ' + str(1000 * mpjpe.mean()))
        #print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
        #print('AUC mean: ' + str( np.array(auc_3d).mean()))
        print('AUC: ' + str(auc ))


def main():
    """
        @date:    2021.03.09
        @readme:  visualization of attention map.
    """
    opt = BaseOptions().parse()
    Trainer(opt).eval(eval_dataset=opt.eval_dataset)

def test():
    opt = BaseOptions().parse()
    Trainer(opt).test(test_dir='experiments/0219/img')
    
def demo():
    opt = BaseOptions().parse()
    eval_set = opt.eval_dataset
    Trainer(opt).demo(eval_set=eval_set)

if __name__ == '__main__':
#     main()
#     test()
    demo()