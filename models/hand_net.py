# coding: UTF-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import resnet
from . import vision_transformer
from . import vision_transformer_attn
from . import vision_performer
from . import hrnet
from . import inception
import numpy as np
import random
from torch import autograd

# # debug
# from models import vision_transformer
# from models import vision_performer
# from models import resnet

def get_model(arch):
    if hasattr(resnet, arch):
        network = getattr(resnet, arch)
        return network(pretrained=True, num_classes=512)
    else:
        raise ValueError("Invalid Backbone Architecture")

class H3DWEncoder(nn.Module):
    def __init__(self, opt, mean_params):
        super(H3DWEncoder, self).__init__()
        #self.two_branch = opt.two_branch
        self.mean_params = mean_params.clone().cuda()
        #self.opt = opt
        self.total_params_dim=61
        self.main_encoder='resnet50'

        relu = nn.ReLU(inplace=False)
        fc2  = nn.Linear(1024, 1024)
        regressor = nn.Linear(1024 + self.total_params_dim, self.total_params_dim)

        feat_encoder = [relu, fc2, relu]
        regressor = [regressor, ]
        self.feat_encoder = nn.Sequential(*feat_encoder)
        self.regressor = nn.Sequential(*regressor)

        self.main_encoder = get_model(self.main_encoder)


    def forward(self, main_input):
        main_feat,_,_,_,_ = self.main_encoder(main_input)
        feat = self.feat_encoder(main_feat)

        pred_params = self.mean_params
        for i in range(3):
            input_feat = torch.cat([feat, pred_params], dim=1)
            output = self.regressor(input_feat)
            pred_params = pred_params + output
        return feat, pred_params  #add feat


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_model(arch):
    if hasattr(resnet, arch):
        network = getattr(resnet, arch)
        return network(pretrained=True, num_classes=512)
    else:
        raise ValueError("Invalid Backbone Architecture")


class EncoderTransformerInception(nn.Module):
    def __init__(self, opt, mean_params):
        super(EncoderTransformerInception, self).__init__()
        self.mean_params = mean_params.clone().cuda()
        self.total_params_dim=61
        self.main_encoder=inception.Inception3(aux_logits=False)
        
        depth = opt.vit_depth
        heads = opt.vit_heads
        
        self.conv1x1_channel_reduction = nn.Conv2d(192, 128, 3, 2, 3, bias=False)
        self.transformer = vision_transformer.Transformer(dim=196, depth=depth, heads=heads, dim_head=64, mlp_dim=392, dropout=0.0)
            
        self.iteration = opt.iteration
        self.regressor = nn.Sequential(
            nn.Linear(196+self.total_params_dim, self.total_params_dim)
        )
        
        self.pos_embed = opt.pos_embed
        self.positionalEncoding = PositionalEncoding(196,
                                                     max_len=128)
        mask_token = torch.randn(1, 1, 196)
        self.mask_token = nn.Parameter(mask_token)
        self.mask_rate = opt.mask_rate

    def forward(self, main_input):
        main_feat = self.main_encoder(main_input)
        # main_feat [bs, 1024]
        # x1        [bs, 256, 56, 56]
        # x2        [bs, 512, 28, 28]
        # x3        [bs, 1024, 14, 14]
        # x4        [bs, 2048, 7, 7]
        pred_params = self.mean_params
        pred_params = pred_params.unsqueeze(1)
        pred_params = pred_params.repeat(main_feat.size(0), 128, 1)        
            
        feat = main_feat.view(main_feat.size(0), 192, 24, 24)
        feat = self.conv1x1_channel_reduction(feat)
        feat = feat.view(feat.size(0), 128, -1)
        
        if self.pos_embed:
            feat = self.positionalEncoding(feat)
        if self.mask_rate >= 0.1 and self.mask_rate <= 0.9:
            masked = list(range(128))
            random.shuffle(masked)
            masked = masked[:int(self.mask_rate*128)]
            feat[:, masked, :] = self.mask_token
        
        feat = self.transformer(feat, None)
        feat = feat.mean(dim=1)
        
        pred_params = self.mean_params
        pred_params = pred_params.repeat(feat.size(0), 1)
        
        for _ in range(self.iteration):
            input_feat = torch.cat([feat, pred_params], dim=-1)
            out = self.regressor(input_feat)
            pred_params = pred_params + out
        
        return pred_params
    
    
    
class EncoderTransformerHRNet(nn.Module):
    def __init__(self, opt, mean_params):
        super(EncoderTransformerHRNet, self).__init__()
        self.mean_params = mean_params.clone().cuda()
        self.total_params_dim=61
        self.main_encoder=hrnet.HRNet(c=24, nof_joints=128, bn_momentum=0.1)
        
        depth = opt.vit_depth
        heads = opt.vit_heads
        
        self.conv1x1_channel_reduction = nn.Conv2d(512, 128, 3, 2, 1, bias=False)
        self.transformer = vision_transformer.Transformer(dim=196, depth=depth, heads=heads, dim_head=64, mlp_dim=392, dropout=0.0)
            
        self.iteration = opt.iteration
        self.regressor = nn.Sequential(
            nn.Linear(196+self.total_params_dim, self.total_params_dim)
        )
        
        self.pos_embed = opt.pos_embed
        self.positionalEncoding = PositionalEncoding(196,
                                                     max_len=128)
        mask_token = torch.randn(1, 1, 196)
        self.mask_token = nn.Parameter(mask_token)
        self.mask_rate = opt.mask_rate
        

    def forward(self, main_input):
        main_feat = self.main_encoder(main_input)
        # main_feat [bs, 1024]
        # x1        [bs, 256, 56, 56]
        # x2        [bs, 512, 28, 28]
        # x3        [bs, 1024, 14, 14]
        # x4        [bs, 2048, 7, 7]
        pred_params = self.mean_params
        pred_params = pred_params.unsqueeze(1)
        pred_params = pred_params.repeat(main_feat.size(0), 128, 1)        
            
        feat = main_feat.view(main_feat.size(0), 512, 28, 28)
        feat = self.conv1x1_channel_reduction(feat)
        feat = feat.view(feat.size(0), 128, -1)
        print(feat.shape)
        
        # 1) positional embedding.(https://github.com/zankner/IMask/issues/1)
        if self.pos_embed:
            feat = self.positionalEncoding(feat)
        # 2) masking mechanism
        if self.mask_rate >= 0.1 and self.mask_rate <= 0.9:
            masked = list(range(128))
            random.shuffle(masked)
            masked = masked[:int(self.mask_rate*128)]
            feat[:, masked, :] = self.mask_token
        
        feat = self.transformer(feat, None)
        feat = feat.mean(dim=1)
        
        pred_params = self.mean_params
        pred_params = pred_params.repeat(feat.size(0), 1)
        
        for _ in range(self.iteration):
            input_feat = torch.cat([feat, pred_params], dim=-1)
            out = self.regressor(input_feat)
            pred_params = pred_params + out
        
        return pred_params
    

class EncoderTransformerCoarse(nn.Module):
    """
        @date:   2021.02.06 week5
    """
    def __init__(self, opt, mean_params):
        super(EncoderTransformerCoarse, self).__init__()
        self.mean_params = mean_params.clone().cuda()
        self.main_encoder='resnet50'
        
        heads = opt.vit_heads
        self.pl = opt.pl_reg
        
#         if opt.feature == 'coarse':
        self.full_content = 21
        self.conv1x1_channel_reduction = nn.Conv2d(512, 21, 1, 1, 0, bias=False)
        
        # normal
#         self.transformer = vision_transformer.Transformer(dim=784, depth=3, heads=heads, dim_head=64, mlp_dim=392, dropout=0.0)
        
        # attn.
        self.transformer = vision_transformer_attn.Transformer(dim=784, depth=3, heads=8, dim_head=64, mlp_dim=392, dropout=0.0)

        self.main_encoder = get_model(self.main_encoder)
        self.iteration = opt.iteration
        
        # 1) positional embedding.(https://github.com/zankner/IMask/issues/1)
        self.pos_embed = opt.pos_embed
        if self.pos_embed is True:
            print("Position Encoding open")
        else:
            print("Position Encoding close")
        self.positionalEncoding = PositionalEncoding(784,
                                                     max_len=21)
        
#         self.positionalEncodingViT = torch.nn.Parameter(torch.randn(1, 21, 784))
        
        # 2) masking mechanism
        mask_token = torch.randn(1, 1, 784)
        self.mask_token = nn.Parameter(mask_token)
        self.mask_rate = opt.mask_rate
        
        # 3) regressor
#         self.iteration = opt.iteration
        self.regressor = nn.Linear(1024+3, 3)

    def forward(self, main_input):
        main_feat,x1,x2,x3,x4 = self.main_encoder(main_input)
        # main_feat [bs, 1024]
        # x1        [bs, 256, 56, 56]
        # x2        [bs, 512, 28, 28]  (path regularization.)
        # x3        [bs, 1024, 14, 14]
        # x4        [bs, 2048, 7, 7]
        # x2 -> channel reduction, feed regressor.
        
        feat_visual = self.conv1x1_channel_reduction(x2)
        feat = feat_visual.view(feat_visual.size(0), 21, -1)
        
        # 1) positional embedding.(https://github.com/zankner/IMask/issues/1)
        if self.pos_embed:
            feat = self.positionalEncoding(feat)  # 0207 exp
#             feat += self.positionalEncodingViT      # 0211 exp
        # 2) mask Timestep wise content
        if self.mask_rate >= 0.1 and self.mask_rate <= 0.9:
            masked = list(range(self.full_content))
            random.shuffle(masked)
            masked = masked[:int(self.mask_rate*self.full_content)]
            feat[:, masked, :] = self.mask_token
        
        # normal
        # feat_out = self.transformer(feat, None)  # coarse: [bs, 21, 3]
        # visualize attn
        feat_out, attn = self.transformer(feat, None)  # coarse: [bs, 21, 3]
        feat_out = feat_out.view(feat_out.size(0), -1)
        
        mean_params = self.mean_params
        mean_params = mean_params.repeat(x1.size(0), 1)
        
        # todo: 2021.02.06  hand_mano.mesh_mu
        pred_params = mean_params.clone()
        pred_params[:, 3:] = pred_params[:, 3:] + feat_out
        
        cameras = self.regressor(torch.cat((main_feat, pred_params[:, :3]), dim=1))
            
        pred_3d = pred_params[:, 3: (3 + 63)].view(-1, 21, 3)
        root = pred_3d[:,1].clone().unsqueeze(1)
        pred_3d -= root
        
        pred_params[:, 3:] = pred_3d.view(-1, 63)
        pred_params[:, :3] = cameras
        
#         return pred_params, feat_visual
        # with pose length reg
        if self.pl:
            return pred_params, feat_visual, attn, autograd.grad(torch.sum(feat_out), feat_visual, retain_graph=True)[0]
        else:
            return pred_params, feat_visual, attn



class EncoderTransformer(nn.Module):
    """
        @date:   2021.03.09 week10
    """
    def __init__(self, opt, mean_params):
        super(EncoderTransformer, self).__init__()
        self.mean_params = mean_params.clone().cuda()
        self.main_encoder='resnet50'
        
        heads = opt.vit_heads
        self.pl = opt.pl_reg
        
#         if opt.feature == 'coarse':
        self.full_content = 21
        self.conv1x1_channel_reduction = nn.Conv2d(512, 21, 1, 1, 0, bias=False)
        
        self.transformer = vision_transformer.Transformer(dim=784, depth=3, heads=heads, dim_head=64, mlp_dim=392, dropout=0.0)
        
#         self.transformer = vision_transformer_attn.Transformer(dim=784, depth=3, heads=8, dim_head=64, mlp_dim=392, dropout=0.0)
        
        self.main_encoder = get_model(self.main_encoder)
        self.iteration = opt.iteration
        
        self.pos_embed = opt.pos_embed
        if self.pos_embed is True:
            print("Position Encoding open")
        else:
            print("Position Encoding close")
        self.positionalEncoding = PositionalEncoding(784,
                                                     max_len=21)
        
#         self.positionalEncodingViT = torch.nn.Parameter(torch.randn(1, 21, 784))
        
        mask_token = torch.randn(1, 1, 784)
        self.mask_token = nn.Parameter(mask_token)
        self.mask_rate = opt.mask_rate
        
        self.iteration = opt.iteration
        self.regressor = nn.Linear(1024+66, 66)

    def forward(self, main_input):
        main_feat,x1,x2,x3,x4 = self.main_encoder(main_input)
        # main_feat [bs, 1024]
        # x1        [bs, 256, 56, 56]
        # x2        [bs, 512, 28, 28]  (path regularization.)
        # x3        [bs, 1024, 14, 14]
        # x4        [bs, 2048, 7, 7]
        
        feat_visual = self.conv1x1_channel_reduction(x2)
        feat = feat_visual.view(feat_visual.size(0), 21, -1)
        
        if self.pos_embed:
            feat = self.positionalEncoding(feat)  
#             feat += self.positionalEncodingViT     
        if self.mask_rate >= 0.1 and self.mask_rate <= 0.9:
            masked = list(range(self.full_content))
            random.shuffle(masked)
            masked = masked[:int(self.mask_rate*self.full_content)]
            feat[:, masked, :] = self.mask_token
        
        feat_out = self.transformer(feat, None)  # coarse: [bs, 21, 3]
#         feat, attn = self.transformer(feat, None)  # coarse: [bs, 21, 3]
        feat_out = feat_out.view(feat_out.size(0), -1)
        
        mean_params = self.mean_params
        mean_params = mean_params.repeat(x1.size(0), 1)
        
        pred_params = mean_params.clone()
        pred_params[:, 3:] = pred_params[:, 3:] + feat_out
        
        for i in range(self.iteration):
            output = self.regressor(torch.cat((main_feat, pred_params), dim=1))
            pred_params = pred_params + output
            
        pred_3d = pred_params[:, 3: (3 + 63)].view(-1, 21, 3)
        root = pred_3d[:,1].clone().unsqueeze(1)
        pred_3d -= root
        
        pred_params[:, 3:] = pred_3d.view(-1, 63)
        
        if self.pl:
            return pred_params, feat_visual, autograd.grad(torch.sum(feat_out), feat_visual, retain_graph=True)[0]
        else:
            return pred_params, feat_visual

    
if __name__ == "__main__":
    mean_params = torch.randn(1, 66)
    enc = EncoderTransformer(mean_params)
    ins = torch.randn(1,3,224,224)
    enc(ins)