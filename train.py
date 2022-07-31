# -*- coding: utf-8 -*-
"""
    @date:      2021.02.12 week6
    @update:    1) original positional encoding is more suitable for our task than ViT version.
                2) HMR(CVPR2018, Angjoo Kanazawa, Michael J. Black) for regressor: which solves hard to convergence while iterations goes wild.
"""
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from models.hand_net import EncoderTransformer
import numpy as np
import pickle
import os
import os.path as osp
from torch.nn.parallel import DistributedDataParallel

from config import BaseOptions
from dataset.MultiDataset import concat_dataset
import time
from data_utils.draw_3d_joints import debug_pred_gt
import data_utils.general_utils as gnu

from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR, ExponentialLR

class Trainer():
    def __init__(self, opt):
        self.opt=opt
        self.batch_size=opt.batch_size
        self.lr=opt.lr
        self.resume=opt.resume
        self.epoches=opt.epoch
        self.train_loader=concat_dataset(self.batch_size,opt)
        self.debug=opt.debug
        self.debug_img_name = opt.debug_img
        
        self.pl = opt.pl_reg
        if self.pl:
            print("with pose length reg")
        else:
            print("no pose length reg")
        
        self.load_mano_mean(opt.outside)
        
        #loss weight:
        self.l_weight_3d = opt.l_weight_3d
        self.l_weight_2d = opt.l_weight_2d
        
        # 网络.
        if opt.net == 'reg_performer':
            pass
        elif opt.net == 'reg_transformer':
            print("[iccv2021 scat] Transformer regressor...")
            self.net = EncoderTransformer(opt, self.mean_params).cuda()

        # optimizer.
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        scheduler_steplr = StepLR(self.optimizer, step_size=10, gamma=1)
        print("batch num", len(self.train_loader))
        self.scheduler_warmup = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=15, after_scheduler=scheduler_steplr)

        self.checkpoint_path = opt.checkpoint_hand
        if self.resume:
            checkpoint_path = self.checkpoint_path
            if not osp.exists(checkpoint_path):
                print(f"Error: {checkpoint_path} does not exists, Start from Scratch...")
                self.success_load = False
            else:
                saved_weights = torch.load(checkpoint_path)
                self.net.load_state_dict(saved_weights,strict=False)
                print('Checkpoint loaded from: '+checkpoint_path)
                self.success_load = True
    
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
            # back of hand(local), blender check hand_mean.
            self.local_tree = [188, 142, 87, 290, 216, 316, 402, 200, 585, 630, 285, 473, 513, 88, 249, 702, 329, 439, 668, 550, 740]
        else:
            # palm of hand(local).
            self.local_tree = [35, 168, 47, 337, 283, 353, 449, 591, 599, 637, 139, 467, 560, 5, 121, 707, 329, 439, 668, 550, 740]
        
        # blender index start from 1.
        self.local_tree = list(map(lambda x: x-1, self.local_tree))
        
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
    
    def project_2d(self,joints_2d):
        return joints_2d*112+112
    
    def train(self):
        
        self.optimizer.zero_grad()
        self.optimizer.step()
        
        for epoch in range(self.epoches):  # loop over the dataset multiple times
            t0=time.time()
            running_loss = 0.0
            loss=0.0
            loss_3d=0.0
            loss_2d=0.0
            loss_pl=0.0  # pose length reg
            self.scheduler_warmup.step(epoch+1)
            t6=time.time()
            for i, datas in enumerate(self.train_loader):
                t1 = time.time()
                for data in datas:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    inputs, labels = data
                    
                    # 2021.01.31 skip empty data.
                    content = abs(inputs.sum(dim=[1,2,3]))
                    idxs = torch.where(abs(content - 224*224*3) > 2000, torch.ones_like(content), torch.zeros_like(content))
                    idx_lst = idxs * torch.arange(self.batch_size)
                    idx_lst = list(map(lambda x: int(x), idx_lst))
                    idx_lst = [_ for idx, _ in enumerate(idx_lst) if _ == idx]
                    
                    inputs = inputs[idx_lst, ...]
                    labels = labels[idx_lst, ...]
                    
                    inputs = inputs.to(device).float()
                    labels = labels.to(device).float()
                    self.optimizer.zero_grad()

                    #print('Get output from input ...')
                    t2 = time.time()
                    if self.pl:
                        outputs, _, pl_term = self.net(inputs)
                    else:
                        outputs, _ = self.net(inputs)
                        
                    t3 = time.time()
                    #get seperate outputs
                    cam_dim=3
                    pred_cam_params = outputs[:, :cam_dim]
                    pred_joints_3d = outputs[:, 3:66].view(-1,21,3)

                    # generate predicted joints 2d
                    pred_joints_2d = self.batch_orth_proj_idrot(pred_joints_3d, pred_cam_params)
                    pred_joints_2d = self.project_2d(pred_joints_2d)
                    
                    pred_joints_3d=pred_joints_3d.view(-1,63)
                    pred_joints_2d=pred_joints_2d.view(-1,42)
                    
                    #pose length reg loss
#                     print(pl_term.shape)
                    if self.pl:
                        pl_lengths = torch.sum(torch.square(pl_term), dim=[2,3]).mean(dim=[1]).sqrt()
                        pl_mean_var = 0.0
                        pl_mean = pl_mean_var + 0.01 * (torch.mean(pl_lengths) - pl_mean_var)
                        pl_mean_var = pl_mean
                        l_pl = torch.square(pl_lengths - pl_mean).mean()
                    else:
                        l_pl = 0.0
                    
                    #seperate loss
                    if labels.size()[1]==105:    #MTC, RHD, STB only have 3d & 2d joints gt
                        gt_joints_3d=labels[:,:63]
                        gt_joints_2d=labels[:,63:]
                        l_3d = nn.MSELoss()(pred_joints_3d,gt_joints_3d)
                        l_2d = nn.L1Loss()(pred_joints_2d,gt_joints_2d)
                    else:                       # freihand, ho3d have pose, 3d & 2d joints gt
                        gt_pose=labels[:,3:51]
                        gt_joints_3d=labels[:,61:(61+63)]
                        gt_joints_2d=labels[:,(61+63):]
                        l_3d = nn.MSELoss()(pred_joints_3d,gt_joints_3d)
                        l_2d = nn.L1Loss()(pred_joints_2d,gt_joints_2d)
                    
                    if self.pl:
                        loss= self.l_weight_3d* l_3d + self.l_weight_2d * l_2d + 10 * l_pl
                    else:
                        loss= self.l_weight_3d* l_3d + self.l_weight_2d * l_2d
                    
                    t4 = time.time()
                    loss.backward()
                    
                    t5 = time.time()
                    self.optimizer.step()
                    
                    if self.debug and i % 100 == 0:
                        print('==== Visualize ====')    
                        img = inputs[0]
                        img=img.permute(1,2,0).cpu().numpy()
                        image=np.clip(img*127.5+127.5,0,255).astype(np.uint8)

                        gt_joints_3d=gt_joints_3d[0].cpu().numpy().reshape(21,3)
                        gt_joints_2d=gt_joints_2d[0].cpu().numpy().reshape(21,2)
                        pred_joints_3d=pred_joints_3d[0].cpu().detach().numpy().reshape(21,3)
                        pred_joints_2d=pred_joints_2d[0].cpu().detach().numpy().reshape(21,2)

                        debug_pred_gt(image,gt_joints_2d,gt_joints_3d,pred_joints_2d,pred_joints_3d,self.debug_img_name)

                # print statistics
                if loss != 0.0:
                    running_loss += loss.item()
                    loss_3d += self.l_weight_3d *l_3d
                    loss_2d += self.l_weight_2d *l_2d
                    loss_pl += 10*l_pl
                    if i % 10 == 0:    #1999:    # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f, 3d loss: %.3f, 2d loss: %.3f, pose length reg: %.3f' %
                              (epoch + 1, i + 1, running_loss/10,loss_3d/10,loss_2d/10,loss_pl))
                        #writer.add_scalar('training loss',running_loss / 1000,epoch * len(trainloader) + i)
                        running_loss = 0.0
                        loss_3d=0.0
                        loss_2d=0.0
            if epoch%10==0:
                if not os.path.isdir(self.opt.checkpoint_folder):
                    os.mkdir(self.opt.checkpoint_folder)
                    print('Create folder :'+self.opt.checkpoint_folder)
                PATH=osp.join(self.opt.checkpoint_folder,'hand_net.pth')
                torch.save(self.net.state_dict(),PATH)

        print('Finished Training')
        PATH = osp.join(self.opt.checkpoint_folder,'./hand_net_final.pth')
        torch.save(self.net.state_dict(), PATH)

def main():
    opt = BaseOptions().parse()
    Trainer(opt).train()

if __name__ == '__main__':
    main()
