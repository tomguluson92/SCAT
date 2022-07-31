# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

MIN_NUM_PATCHES = 16

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, attn, **kwargs):
        return self.fn(x, **kwargs) + x, attn

class PreNormAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, **kwargs):
        return self.norm(x)
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None):
        super().__init__()
        if out_dim is None:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim // 2)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3)
            )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out, attn

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            if _ == depth-1:
                self.layers.append(nn.ModuleList([
                    Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    PreNormAttn(dim),
                    FeedForward(dim, (dim * 3) // 4, out_dim=3)
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    PreNormAttn(dim),
                    PreNorm(dim, FeedForward(dim, (dim * 3) // 4))
                ]))
                dim = dim // 2
    def forward(self, x, mask = None):
        for attention, pren, ff in self.layers:
            x1, attn = attention(x, mask = mask)
            x = pren(x1) + x
            x = ff(x)
#             print(x.shape)
#             print(attn.shape)
        return x, attn

class ViT(nn.Module):
    def __init__(self, *, opt, mean_params, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'mean', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(dim+num_classes, num_classes)
        )
        
        self.iteration = opt.iteration
        self.mean_params = mean_params.clone().cuda()

    def forward(self, img, mask = None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        feat = x.mean(dim = 1)
        pred_params = self.mean_params
        pred_params = pred_params.repeat(x.size(0), 1)
        
        for _ in range(self.iteration):
            input_feat = torch.cat([feat, pred_params], dim=1)
            output = self.head(input_feat)
            pred_params = pred_params + output
        return pred_params