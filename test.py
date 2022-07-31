# -*- coding: utf-8 -*-
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from models.hand_net import H3DWEncoder
import os
import numpy as np
import pickle
import smplx
import os.path as osp
from torch.nn.parallel import DistributedDataParallel
from models.mano import ManoHand,rot_pose_beta_to_mesh
from config import BaseOptions
#from dataset.MultiDataset import concat_dataset
import torchvision.transforms as transforms
from PIL import Image
from dataset.inference import Inference
from torch.utils.data import DataLoader
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

def center_crop_resize(img):
    #img=img[:, 80:560, :]
    width, height = img.size
    if width>height:
        top=0
        bottom=height
        left=(width-height)//2
        right=width-(width-height)//2
    else:
        left=0
        right=width
        top=(height-width)//2
        bottom=height-(height-width)//2
    img=img.crop((left, top, right, bottom))

    new_width,new_height=img.size
    img=img.resize((224,224))
    dx=np.array([-left, 0])
    dy=np.array([0, -top])
    scale=224/new_width
    return img

def convert_bbox_to_oriIm(data3D, boxScale_o2n, bboxTopLeft, imgSizeW, imgSizeH):
    data3D = data3D.copy()
    #data3D=data3D.cpu().detach().numpy()
    resnet_input_size_half = 224 *0.5
    imgSize = np.array([imgSizeW,imgSizeH])

    data3D /= boxScale_o2n

    if not isinstance(bboxTopLeft, np.ndarray):
        assert isinstance(bboxTopLeft, tuple)
        assert len(bboxTopLeft) == 2
        bboxTopLeft = np.array(bboxTopLeft)
    
    data3D[:,:,:2] += (bboxTopLeft + resnet_input_size_half/boxScale_o2n)
    #data3D = transforms.ToTensor()(data3D)#.unsqueeze_(0)
    #data3D = data3D.to(device).float()
    return data3D

def convert_smpl_to_bbox(data3D, scale, trans, bAppTransFirst=False):
    #data3D = data3D.copy()
    print(data3D.shape,scale.shape,trans.shape)
    resnet_input_size_half = 224 *0.5
    if bAppTransFirst:      # Hand model
        data3D[:,:,0:2] += trans
        data3D *= scale   # apply scaling
    else:
        data3D *= scale # apply scaling
        data3D[:,0:2] += trans
    
    data3D*= resnet_input_size_half # 112 is originated from hrm's input size (224,24)
    # data3D[:,:2]*= resnet_input_size_half # 112 is originated from hrm's input size (224,24)
    return data3D

def load_pkl(pkl_file, res_list=None):
    assert pkl_file.endswith(".pkl")
    with open(pkl_file, 'rb') as in_f:
        try:
            data = pickle.load(in_f)
        except UnicodeDecodeError:
            in_f.seek(0)
            data = pickle.load(in_f, encoding='latin1')
    return data


def extract_hand_output(output, hand_type, hand_info, top_finger_joints_type='ave', use_cuda=True):
    assert hand_type in ['left', 'right']

    if hand_type == 'left':
        wrist_idx, hand_start_idx, middle_finger_idx = 20, 25, 28
    else:
        wrist_idx, hand_start_idx, middle_finger_idx = 21, 40, 43

    vertices = output.vertices
    joints = output.joints
    vertices_shift = vertices - joints[:, hand_start_idx:hand_start_idx + 1, :]

    hand_verts_idx = torch.Tensor(hand_info[f'{hand_type}_hand_verts_idx']).long()
    if use_cuda:
        hand_verts_idx = hand_verts_idx.cuda()

    hand_verts = vertices[:, hand_verts_idx, :]
    hand_verts_shift = hand_verts - joints[:, hand_start_idx:hand_start_idx + 1, :]

    hand_joints = torch.cat((joints[:, wrist_idx:wrist_idx + 1, :],
                             joints[:, hand_start_idx:hand_start_idx + 15, :]), dim=1)

    # add top hand joints
    if len(top_finger_joints_type) > 0:
        if top_finger_joints_type in ['long', 'manual']:
            key = f'{hand_type}_top_finger_{top_finger_joints_type}_vert_idx'
            top_joint_vert_idx = hand_info[key]
            hand_joints = torch.cat((hand_joints, vertices[:, top_joint_vert_idx, :]), dim=1)
        else:
            assert top_finger_joints_type == 'ave'
            key1 = f'{hand_type}_top_finger_{top_finger_joints_type}_vert_idx'
            key2 = f'{hand_type}_top_finger_{top_finger_joints_type}_vert_weight'
            top_joint_vert_idxs = hand_info[key1]
            top_joint_vert_weight = hand_info[key2]
            bs = vertices.size(0)

            for top_joint_id, selected_verts in enumerate(top_joint_vert_idxs):
                top_finger_vert_idx = hand_verts_idx[np.array(selected_verts)]
                top_finger_verts = vertices[:, top_finger_vert_idx]
                # weights = torch.from_numpy(np.repeat(top_joint_vert_weight[top_joint_id]).reshape(1, -1, 1)
                weights = top_joint_vert_weight[top_joint_id].reshape(1, -1, 1)
                weights = np.repeat(weights, bs, axis=0)
                weights = torch.from_numpy(weights)
                if use_cuda:
                    weights = weights.cuda()
                top_joint = torch.sum((weights * top_finger_verts), dim=1).view(bs, 1, 3)
                hand_joints = torch.cat((hand_joints, top_joint), dim=1)

    hand_joints_shift = hand_joints - joints[:, hand_start_idx:hand_start_idx + 1, :]

    output = dict(
        wrist_idx=wrist_idx,
        hand_start_idx=hand_start_idx,
        middle_finger_idx=middle_finger_idx,
        vertices_shift=vertices_shift,
        hand_vertices=hand_verts,
        hand_vertices_shift=hand_verts_shift,
        hand_joints=hand_joints,
        hand_joints_shift=hand_joints_shift
    )
    return output


class Trainer():
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = 64
        self.lr = opt.lr
        self.resume = opt.resume

        self.top_finger_joints_type = 'ave'
        #self.train_loader = concat_dataset(self.batch_size)

        self.mean_mano_params = opt.mean_mano_param
        self.load_params()
        # set differential SMPL (implemented with pytorch) and smpl_renderer
        # smplx_model_path = osp.join(opt.model_root, opt.smplx_model_file)
        smplx_model_path = opt.smplx_model_path
        self.smplx = smplx.create(
            smplx_model_path,
            model_type="smplx",
            batch_size=self.batch_size,
            gender='neutral',
            num_betas=10,
            use_pca=False,
            ext='pkl').cuda()

        self.net = H3DWEncoder(opt, self.mean_params).cuda()

        # define loss
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        self.checkpoint_path = opt.checkpoint_hand
        if self.resume:
            checkpoint_path = self.checkpoint_path
            if not osp.exists(checkpoint_path):
                print(f"Error: {checkpoint_path} does not exists")
                self.success_load = False
            else:
                if self.opt.dist:
                    self.net.module.load_state_dict(torch.load(
                        checkpoint_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))
                else:
                    saved_weights = torch.load(checkpoint_path)
                    self.net.load_state_dict(saved_weights)
                self.success_load = True

    def load_params(self):
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
        mean_params = np.repeat(mean_params, self.batch_size, axis=0)
        self.mean_params = torch.from_numpy(mean_params).float()
        self.mean_params.requires_grad = False
        # define global rotation
        self.global_orient = torch.zeros((self.batch_size, 3), dtype=torch.float32).cuda()
        # self.global_orient[:, 0] = np.pi
        self.global_orient.requires_grad = False

        # load smplx-hand faces
        hand_info_file = self.opt.smplx_hand_info_file
        self.hand_info = load_pkl(hand_info_file)

    def get_smplx_output(self, pose_params, shape_params=None):
        hand_rotation = pose_params[:, :3]
        hand_pose = pose_params[:, 3:]
        body_pose = torch.zeros((self.batch_size, 63)).float().cuda()
        body_pose[:, 60:] = hand_rotation  # set right hand rotation

        output = self.smplx(
            global_orient=self.global_orient,
            body_pose=body_pose,
            right_hand_pose=hand_pose,
            betas=shape_params,
            return_verts=True)

        hand_output = extract_hand_output(
            output,
            hand_type='right',
            hand_info=self.hand_info,
            top_finger_joints_type=self.top_finger_joints_type,
            use_cuda=True)

        pred_verts = hand_output['vertices_shift']
        pred_joints_3d = hand_output['hand_joints_shift']
        return pred_verts, pred_joints_3d

    def batch_orth_proj_idrot(self, X, camera):
        # camera is (batchSize, 1, 3)
        camera = camera.view(-1, 1, 3)
        X_trans = X[:, :, :2] + camera[:, :, 1:]#-camera[:,:,1:]
        res = camera[:, :, 0] * X_trans.view(X_trans.size(0), -1)#/camera[:,:,0]
        return res.view(X_trans.size(0), X_trans.size(1), -1)
    def project_2d(self,joints_2d):
        return joints_2d*112+112
    
    def test(self):

        # for epoch in range(100):  # loop over the dataset multiple times
        #
        #     running_loss = 0.0
        #     for i, (inpusts, labels) in enumerate(self.train_loader):
        #
        #         # print(type(inputs))
        #         # print(inputs.size())
        saved_weights = torch.load('../frankmocap-master/extra_data/hand_module/pretrained_weights/pose_shape_best.pth')
        #saved_weights = torch.load('../experiments/3DHand/checkpoints/hand_net_3d_2set_final.pth')
        #saved_weights = torch.load('../experiments/3DHand/1222-2set/checkpoints/hand_net_final.pth')
        
        self.net.load_state_dict(saved_weights)
        self.net.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        image_folder = './rgb/'
        frames=[]
        for idx,img_file in enumerate(sorted(os.listdir(image_folder))):
            frames.append(idx)
        frames=np.array(frames)
        bboxes = joints2d = None


        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=1.1,
        )

        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if joints2d is not None else False

        dataloader = DataLoader(dataset, batch_size=64, num_workers=16,drop_last=True)
        
        with torch.no_grad():

            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, gt_joints2d, pred_joints2d ,gt_joints3d = [], [], [], [], [], [], [], []
            pred_feature=[]

            for batch in dataloader:
                has_keypoints=True
                if has_keypoints:
                    batch, j2d,j3d = batch
                    gt_joints2d.extend(j2d.cpu().numpy().reshape(-1, 21, 2))
                    gt_joints3d.extend(j3d.cpu().numpy().reshape(-1, 21, 3))
                
                #batch = batch.unsqueeze(0)
                
                batch = batch.to(device)
                batch_size=1
                seqlen=64
                #batch_size, seqlen = batch.shape[:2]
                print(batch_size,seqlen)
                #batch = batch.reshape(batch_size* seqlen, -1)
                print(batch.shape)
                features, outputs = self.net(batch)

                # get seperate outputs
                cam_dim = 3
                pose_dim = 48
                pred_cam_params = outputs[:, :cam_dim]
                pred_pose_params = outputs[:, cam_dim: (cam_dim + pose_dim)]
                pred_shape_params = outputs[:, (cam_dim + pose_dim):]

                #  get predicted smpl verts and joints,
#                 pred_verts_3d, pred_joints_3d = self.get_smplx_output(pred_pose_params, pred_shape_params)
                # MANO =======================>
                rot = pred_pose_params[:,:3]    
                theta = pred_pose_params[:,3:]
                beta = pred_shape_params
                x3d = rot_pose_beta_to_mesh(rot,theta,beta)
                pred_joints_3d=x3d[:,:21]
        
        
                pred_joints_2d = self.batch_orth_proj_idrot(pred_joints_3d, pred_cam_params)
                pred_joints_2d = self.project_2d(pred_joints_2d)

                pred_cam.append(pred_cam_params.reshape(batch_size * seqlen, -1))
                pred_verts.append(pred_verts_3d.reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(pred_pose_params.reshape(batch_size * seqlen, -1))
                pred_betas.append(pred_shape_params.reshape(batch_size * seqlen, -1))
                pred_joints3d.append(pred_joints_3d.reshape(batch_size * seqlen, -1, 3))
                pred_joints2d.append(pred_joints_2d.reshape(batch_size * seqlen, -1, 2))
                pred_feature.append(features.reshape(batch_size * seqlen, -1))
                
                
            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)
            pred_joints2d = torch.cat(pred_joints2d, dim=0)
            pred_feature = torch.cat(pred_feature, dim=0)

            del batch
            
            pred_cam = pred_cam.cpu().numpy()
            pred_verts = pred_verts.cpu().numpy()
            pred_pose = pred_pose.cpu().numpy()
            pred_betas = pred_betas.cpu().numpy()
            pred_joints3d = pred_joints3d.cpu().numpy()
            pred_joints2d = pred_joints2d.cpu().numpy()
            pred_feature = pred_feature.cpu().numpy()
            
            output_dict = {
                'pred_cam': pred_cam,
                #'orig_cam': orig_cam,
                'verts': pred_verts,
                'pose': pred_pose,
                'betas': pred_betas,
                'joints3d': pred_joints3d,
                'joints2d': pred_joints2d,
                'bboxes': bboxes,
                'frame_ids': frames,
                'feature': pred_feature,
            }
        
        visual_2d_3d=True
        if visual_2d_3d:
            from data_utils.draw_3d_joints import draw_3d_skeleton,plot_2d_hand,plot_3d_hand
            import matplotlib.pyplot as plt
            #output_img_folder=args.img_file.split('/')[-2]
            #output_img_folder ='../output/smplx/'
            output_img_path='../output/smplx/'
            os.makedirs(output_img_path, exist_ok=True)
            print(f'Rendering output video, writing frames to {output_img_path}')
            image_file_names = sorted([
                 os.path.join(image_folder, x)
                 for x in os.listdir(image_folder)
                 if x.endswith('.png') or x.endswith('.jpg')
             ])
            f = open(os.path.join(output_img_path,'smplx_features.txt'),'w')
            f.write('This file contains feature map extracted from frankmocap...')
            f.close()
            for frame_idx in range(output_dict['joints3d'].shape[0]):
                img_fname = image_file_names[frame_idx]
                #img = cv2.imread(img_fname)
                img = Image.open(img_fname).convert('RGB')
                img=center_crop_resize(img)
                pred_joints_2d=output_dict['joints2d'][frame_idx][jointsMapSMPLXToSimple]
                pred_joints_3d=output_dict['joints3d'][frame_idx][jointsMapSMPLXToSimple]
                gt_joints_2d=gt_joints2d[frame_idx][jointsMapSMPLXToSimple]
                gt_joints_3d=gt_joints3d[frame_idx][jointsMapSMPLXToSimple]
                fig = plt.figure()
                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222)
                ax3 = fig.add_subplot(223,projection='3d')
                ax4 = fig.add_subplot(224,projection='3d')
                ax1.imshow(img)
                ax2.imshow(img)
                plot_2d_hand(ax1, gt_joints_2d, order='uv')
                ax1.set_xlabel('ground truth 2d joints', fontsize=10)
                plot_2d_hand(ax2, pred_joints_2d, order='uv')
                ax2.set_xlabel('predict 2d joints', fontsize=10)
                plot_3d_hand(ax3, gt_joints_3d)
                ax3.set_xlabel('ground truth 3d joints', fontsize=10)
                plot_3d_hand(ax4, pred_joints_3d)
                ax4.set_xlabel('predict 3d joints', fontsize=10)
                fig.savefig(os.path.join(output_img_path, f'{frame_idx:06d}.png'))
                f = open(os.path.join(output_img_path,'smplx_features.txt'), 'a')
                f.write(f'\n{frame_idx:06d}\n')
                f.write(str(output_dict['feature'][frame_idx]))
                f.close()
        
                


def main():
    opt = BaseOptions().parse()
    Trainer(opt).test()


if __name__ == '__main__':
    main()
