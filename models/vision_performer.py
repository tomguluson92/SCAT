# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
#### attention with performers : https://arxiv.org/pdf/2009.14794.pdf

class performer_attn_block(nn.Module):
    def __init__(self, emb_s, head, kernel_ratio=0.5, dp_ratio=0.1):
        super().__init__()
        
        emb = emb_s*head
        self.kqv = nn.Linear(emb_s, 3*emb_s)
        self.dp = nn.Dropout(dp_ratio)     
        self.proj = nn.Linear(emb, emb)
        self.emb_s = emb_s
        self.ln1 = nn.LayerNorm(emb)
        self.ln2 = nn.LayerNorm(emb)
        
        self.mlp = nn.Sequential(
            nn.Linear(emb, 4 * emb),
            nn.GELU(),
            nn.Linear(4 * emb, emb),
            nn.Dropout(dp_ratio),
        )

        self.m = int(emb_s * kernel_ratio)
        self.w = nn.Parameter(torch.randn(self.m, emb_s), requires_grad = False)

    def prm_exp(self, x):
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x*x).sum(dim = -1, keepdim = True)).repeat(1, 1, self.m)/2
        wtx = torch.einsum('bti,mi->btm', x, self.w)
        return torch.exp(wtx - xd)/math.sqrt(self.m)

    def forward_single_attn(self, x):

        k, q, v = torch.split(self.kqv(x), self.emb_s, dim = -1)
        kp, qp = self.prm_exp(k), self.prm_exp(q) # (B, T, m), (B, T, m)
        D =  torch.einsum('bti,bi->bt', qp, kp.sum(dim = 1)).unsqueeze(dim = 2) # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v, kp) #(B, emb_s, m)
        #print(kptv.shape, D.shape)
        y = torch.einsum('bti,bni->btn', qp, kptv)/D.repeat(1, 1, self.emb_s) #(B, T, emb_s)/Diag
        return y
    
    def forward_multi_attn(self, x):
        # poor people implements like this : but if you are rich,
        # just fix above single_attn, add one more dim between 0 and 1 for heads,
        # that gives you better gpu usage
        splits = torch.split(x, self.emb_s, dim = -1)
        mha = torch.cat([self.forward_single_attn(tnsr) for tnsr in splits], dim = -1)
        return self.dp(self.proj(mha))
    
    def forward(self, x):
        x = x + self.forward_multi_attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
#         x = x + self.forward_multi_attn(x)
#         x = x + self.mlp(x)
        return x

class ViP(nn.Module):
    def __init__(self, opt, mean_params, image_pix = 64, patch_pix = 4, out_dim=10, emb_s=128, heads=4, depth=3, kernel_ratio=0.5, dropout=0.1):
        super().__init__()
        tokens_cnt = (image_pix//patch_pix)*(image_pix//patch_pix)
        patch_size = 3*patch_pix*patch_pix
        self.pool = "mean"  # 因为我的任务比较特殊, 因此不是只取[B, T, dim]的第1个, 而是平均比较好. 
        
        self.uf = nn.Unfold(kernel_size = [patch_pix, patch_pix], stride = [patch_pix, patch_pix])
        
        emb = emb_s*heads
        
        self.pos_emb = nn.Parameter(torch.zeros(1, tokens_cnt, emb))
        self.dp = nn.Dropout(dropout)
        # self.head的作用和HMR的regressor一模一样.
        self.head = nn.Linear(emb+out_dim, out_dim)
        self.patch_emb = nn.Linear(patch_size, emb)
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb))
        self.mains = nn.Sequential(*[performer_attn_block(emb_s=emb_s, head=heads, kernel_ratio=kernel_ratio, dp_ratio=dropout) for _ in range(depth)])
        self.apply(self._init_weights)
        
        self.iteration = opt.iteration
        self.mean_params = mean_params.clone().cuda()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_emb(self.uf(x).transpose(1, 2)) + self.pos_emb #(B, T, patch_size)
        x = torch.cat([self.cls_token.repeat(b, 1, 1), x], dim = 1)
        x = self.mains(x)
        
        feat = x.mean(dim = 1)
        pred_params = self.mean_params
        pred_params = pred_params.repeat(b, 1)
        
        for _ in range(self.iteration):
            input_feat = torch.cat([feat, pred_params], dim=1)
            output = self.head(input_feat)
            pred_params = pred_params + output
        return pred_params