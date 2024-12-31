import numpy as np
import cv2

import torch
import torch.nn as nn

from .layers import *
from .utils import optical_flow_warping
from .GLA import GLA


class Encoder(nn.Module):

    def __init__(self, in_channels, init_features):
        super(Encoder, self).__init__()
        
        features = init_features
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1_ref = self._block(in_channels, features)
        self.pool1_ref = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gla1 = GLA(patch_size=16, in_chans=features, embed_dim=features)
        
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2_ref = self._block(features, features * 2)
        self.pool2_ref = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gla2 = GLA(patch_size=8, in_chans=features * 2, embed_dim=features * 2)
        
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3_ref = self._block(features * 2, features * 4)
        self.pool3_ref = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gla3 = GLA(patch_size=4, in_chans=features * 4, embed_dim=features * 4)
        
        self.encoder4 = self._block(features * 4, features * 8)
        self.encoder4_ref = self._block(features * 4, features * 8)
        self.gla4 = GLA(patch_size=4, in_chans=features * 8, embed_dim=features * 8)
        
    def _block(self, in_channels, features):
        return nn.Sequential(
            ConvBlock(in_channels, features),
            ConvBlock(features, features),
        )
        
    def forward(self, x, y):
        enc1 = self.encoder1(x)
        enc1_ref = self.encoder1_ref(y)
        enc1_gla = enc1 + self.gla1(enc1, enc1_ref)
        
        enc2 = self.encoder2(self.pool1(enc1_gla))
        enc2_ref = self.encoder2_ref(self.pool1_ref(enc1_ref))
        enc2_gla = enc2 + self.gla2(enc2, enc2_ref)
        
        enc3 = self.encoder3(self.pool2(enc2_gla))
        enc3_ref = self.encoder3_ref(self.pool2_ref(enc2_ref))
        enc3_gla = enc3 + self.gla3(enc3, enc3_ref)
        
        enc4 = self.encoder4(self.pool3(enc3_gla))
        enc4_ref = self.encoder4_ref(self.pool3_ref(enc3_ref))
        enc4_gla = enc4 + self.gla4(enc4, enc4_ref)
        
        return enc4_gla, enc1, enc2, enc3, enc4


class Decoder_block(nn.Module):    
    def __init__(self, in_features, out_features, out_channels, sigmoid=True):
        super(Decoder_block, self).__init__()
        
        self.sigmoid = sigmoid
        
        self.upconv = nn.ConvTranspose2d(
            in_features, out_features, kernel_size=2, stride=2
        )
        self.decoder = nn.Sequential(
            ConvBlock(out_features*2, out_features),
            ConvBlock(out_features, out_features),
        )
        self.side_out = nn.Sequential(
            ConvBlock(out_features, 16),
            Conv3x3(16, out_channels),
        )
    
    def forward(self, dec_feature, enc_feature):
        
        dec_feature = self.upconv(dec_feature)
        dec_feature = torch.cat((dec_feature, enc_feature), dim=1)
        dec_feature = self.decoder(dec_feature)
        output = self.side_out(dec_feature)
        if self.sigmoid:
            output = torch.sigmoid(output)

        return dec_feature, output
    

class MappingBranch(nn.Module):
    def __init__(self, in_channels, features, out_channels):
        super(MappingBranch, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, init_features=features)
              
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(
            ConvBlock(features * 8, features * 16),
            ConvBlock(features * 16, features * 16),
        )

        self.de_block4 = Decoder_block(features * 16, features * 8, out_channels)
        self.de_block3 = Decoder_block(features * 8, features * 4, out_channels)
        self.de_block2 = Decoder_block(features * 4, features * 2, out_channels)
        self.de_block1 = Decoder_block(features * 2, features, out_channels)
        

    def forward(self, current_frame, reference_frame):
        enc_fusion, enc1, enc2, enc3, enc4 = self.encoder(current_frame, reference_frame)
        
        bottleneck = self.bottleneck(self.pool(enc_fusion))
        dec_4, out4 = self.de_block4(bottleneck, enc4)
        dec_3, out3 = self.de_block3(dec_4, enc3)
        dec_2, out2 = self.de_block2(dec_3, enc2)
        _, out1 = self.de_block1(dec_2, enc1)
        
        return out1, [out2, out3, out4]
        

class RawVideoNetwork(nn.Module):
    def __init__(self, args):
        super(RawVideoNetwork, self).__init__()
        self.args = args
        self.optical_flow_warping = optical_flow_warping
        self.confidence_map = nn.Sequential(
            ConvBlock(3 + 3, 16),
            Conv3x3(16, 1),
            nn.Sigmoid()  
        )
        self.conv_raw = Conv1x1(4, 4)
        # self.conv_raw = ConvBlock(4, 4)
        in_channels = 7
        out_channels = 4
        self.mapping_branch = MappingBranch(in_channels=in_channels, features=32, out_channels=out_channels)
    
    def forward(self, from_rgb, to_rgb, from_raw, flow):
        # wapring 
        warping_rgb = self.optical_flow_warping(from_rgb, flow)
        warping_raw = self.optical_flow_warping(from_raw, flow)    

        confidence_map = self.confidence_map(torch.cat([to_rgb, warping_rgb], dim=1))
        raw = self.conv_raw(warping_raw)
        # mapping 
        current_frame = torch.cat([to_rgb, raw * confidence_map], dim=1)
        reference_frame = torch.cat([from_rgb, from_raw], dim=1)

        predict, mid_res = self.mapping_branch(current_frame, reference_frame)
        
        return predict, mid_res



if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from dataset import RawVideoDataset
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from config import args
    import random
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    dataset = RawVideoDataset(root="/mnt/lustrenew/share_new/zhangchen2/Canon5D3/test/", \
                              json_file="/mnt/lustrenew/share_new/zhangchen2/Canon5D3/test/data.json",
                              patch_size=256, raw_bit_depth=14, aug_ratio=1)
    raw, next_raw, rgb, next_rgb, flow, flow_img= dataset[200]
    model = RawVideoNetwork(args)
    cons_raw, cons_rgb = model(rgb.unsqueeze(0), next_rgb.unsqueeze(0), raw.unsqueeze(0), flow.unsqueeze(0))

    # rgb
    rgb = (rgb*255).permute(1,2,0).cpu().numpy().astype('uint8')
    next_rgb = (next_rgb*255).permute(1,2,0).cpu().numpy().astype('uint8')
    cons_rgb = (cons_rgb*255).permute(0,2,3,1).cpu().numpy().astype('uint8')[0]
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    next_rgb = cv2.cvtColor(next_rgb, cv2.COLOR_RGB2BGR)
    cons_rgb = cv2.cvtColor(cons_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite('vision/warp_rgb.png', cons_rgb)
    cv2.imwrite('vision/gt_rgb.png', next_rgb)
    cv2.imwrite('vision/rgb.png', rgb)
    diff = cv2.absdiff(cons_rgb, next_rgb)
    plt.figure()
    plt.imshow(diff, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('vision/diff_rgb.jpg')
    plt.show()

    # raw
    raw = (raw*255).permute(1,2,0).cpu().numpy().astype('uint8')
    next_raw = (next_raw*255).permute(1,2,0).cpu().numpy().astype('uint8')
    cons_raw = (cons_raw*255).permute(0,2,3,1).cpu().numpy().astype('uint8')[0]
    raw = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    next_raw = cv2.cvtColor(next_raw, cv2.COLOR_RGB2BGR)
    cons_raw = cv2.cvtColor(cons_raw, cv2.COLOR_RGB2BGR)

    cv2.imwrite('vision/warp_raw.png', cons_raw)
    cv2.imwrite('vision/gt_raw.png', next_raw)
    cv2.imwrite('vision/raw.png', raw)
    diff = cv2.absdiff(cons_raw, next_raw)
    plt.figure()
    plt.imshow(diff, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('vision/diff_raw.jpg')
    plt.show()
    
    # flow
    flow_img = (flow_img.permute(1,2,0).cpu().numpy()*255).astype("uint8")
    cv2.imwrite('vision/flow.png', flow_img)
    
    confidence_map = np.where(diff >= 10, 0, 255)
    cv2.imwrite('vision/confidence.png', confidence_map)