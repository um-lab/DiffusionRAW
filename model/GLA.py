from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v):
        super(CrossAttention, self).__init__()
        self.dim_in = dim_in
        
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x, y):
        # x, y: batch, n, dim_in
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(y)  # batch, n, dim_k
        v = self.linear_v(y)  # batch, n, dim_v

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n

        y_info = torch.bmm(dist, v)
        
        return x + y_info


class GlobalGuidance(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super(GlobalGuidance, self).__init__()
        self.patch_size = patch_size
        self.pool_x = nn.MaxPool2d(kernel_size=patch_size, stride=patch_size)
        self.pool_y = nn.MaxPool2d(kernel_size=patch_size, stride=patch_size)
        self.proj_x = Mlp(in_chans, out_features=embed_dim)
        self.proj_y = Mlp(in_chans, out_features=embed_dim)

        self.cross_attention = CrossAttention(embed_dim, embed_dim, embed_dim) 
        self.expand = Mlp(embed_dim, out_features=in_chans)
        
    def forward(self, x, y):

        assert x.shape == y.shape, "x and y should have same shape."
        B, C, H, W = x.shape
        x_embed = self.pool_x(x).flatten(2).transpose(1, 2) # [B, Ph*Pw, C]
        y_embed = self.pool_y(y).flatten(2).transpose(1, 2) 
        x_embed = self.proj_x(x_embed)
        y_embed = self.proj_y(y_embed)
        
        x_global = self.cross_attention(x_embed, y_embed)  
        x_global = self.expand(x_global)  # [B, Ph*Pw, C]
        x_global = x_global.permute(0, 2, 1).view(B, C, H // self.patch_size, W// self.patch_size)
        x_global = F.interpolate(x_global, scale_factor=self.patch_size, mode='nearest')
        
        return x_global


class LocalGuidance(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super(LocalGuidance, self).__init__()
        self.patch_size = patch_size
        self.proj_x = Mlp(in_chans, out_features=embed_dim)
        self.proj_y = Mlp(in_chans, out_features=embed_dim)
        self.cross_attention = CrossAttention(embed_dim, embed_dim, embed_dim) 
        self.expand = Mlp(embed_dim, out_features=in_chans)
            
    def forward(self, x, y):

        assert x.shape == y.shape, "x and y should have same shape"
        B, C, H, W = x.shape
        # window_partition
        window_size = self.patch_size
        x_in = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
        x_patches = x_in.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size*window_size,C)
        x_patches = self.proj_x(x_patches)
        
        y = y.view(B, C, H // window_size, window_size, W // window_size, window_size)
        y_patches = y.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size*window_size,C)
        y_patches = self.proj_y(y_patches)
        
        x_local = self.cross_attention(x_patches, y_patches)
        x_local = self.expand(x_local)
        x_local = x_local.view(B, H // window_size, W // window_size, window_size, window_size, C).permute(0,5,1,3,2,4).contiguous()
        x_local = x_local.view(B,C,H,W)
        
        return x_local


class GLA(nn.Module):  

    def __init__(self, patch_size, in_chans, embed_dim):
        super(GLA, self).__init__()
        
        self.globalGuidance = GlobalGuidance(patch_size, in_chans, embed_dim)
        self.localGuidance = LocalGuidance(patch_size, in_chans, embed_dim)
        
    def forward(self, x, y):
        """
        x: current frame
        y: reference frame
        """
        x_global = self.globalGuidance(x, y)
        x_local = self.localGuidance(x, y)
        
        out = x_global + x_local

        return out


if __name__ == '__main__':
    x = torch.randn(2, 256, 32, 32)
    gla = GLA(32, 8, 256, 128)
    gla(x, x)