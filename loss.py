import torch
import torch.nn as nn
import torch.nn.functional as F


def psnr(x, y, max_pixel_value=1.0):
    # Ensure the inputs are of the same shape
    if x.shape != y.shape:
        raise ValueError("Input shapes must be the same.")

    # Calculate the squared error between the images
    mse = F.mse_loss(x, y, reduction='none')
    mse = mse.mean((1,2,3))
    # Calculate the PSNR using the formula: PSNR = 20 * log10(MAX) - 10 * log10(MSE)
    psnr_value = 20 * torch.log10(torch.tensor(max_pixel_value)) - 10 * torch.log10(mse)

    return psnr_value


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
    

class RawVideoLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.l2_loss = nn.MSELoss(reduction='none')
        self.l2_loss_aux = nn.MSELoss()
        self.ssim_loss = SSIM()
        self.ssim_loss_aux = SSIM()
        
    def balance_pixel_num(self, image, loss_values):
        assert torch.min(image) >= 0 and torch.max(image) <= 1, "Image tensor values should be between 0 and 1"
        split_num = 3
        # 定义亮度区间的边界值,计算每个像素所属的亮度区间
        bins = torch.linspace(0, 1, split_num+1).to(loss_values.device)
        binned = torch.zeros_like(image)
        for i, bound in enumerate(bins[:-1]):
            binned[image >= bound] = i
        # binned = torch.bucketize(image, bins)  # need torch_version>1.9
        # 对于每个亮度区间，计算该区间的像素在整个batch中的比例
        weights = torch.zeros_like(image)

        for i in range(0, split_num):
            mask = (binned == i).float()
            total_pixels = torch.prod(torch.tensor(image.shape[1:])) # 单图像素数
            proportion = torch.sum(mask) / (image.shape[0] * total_pixels) # 该区间段像素在batch中的比例
            weights += mask / (split_num*proportion + 1e-6) 
            weights = torch.clamp(weights, 0, 10)
        adjusted_loss_values = loss_values * weights

        return adjusted_loss_values    
    
    
    def process_hard_sample(self, predicts, labels, l2_loss_value):    
        psnr_value = psnr(predicts, labels) # [B, 1]
        
        weights = torch.zeros_like(psnr_value).to(psnr_value.device) + 1
        weights = weights + (psnr_value < 30).float().to(psnr_value.device) * 3
        weights = weights + (psnr_value < 40).float().to(psnr_value.device) * 2
        weights = weights + (psnr_value < 45).float().to(psnr_value.device) * 2
        weights = weights + (psnr_value < 50).float().to(psnr_value.device)
        weights = weights / 9
        b, c, h, w = l2_loss_value.shape
        weights = weights.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, c, h, w)

        return weights * l2_loss_value

        
    def forward(self, predicts, mid_results, labels):
        l2_loss_value = self.l2_loss(predicts, labels) # [b, c, h, w]
        # l2_loss_value = self.balance_pixel_num(labels, l2_loss_value) # [b, c, h, w]
        # l2_loss_value = self.process_hard_sample(predicts, labels, l2_loss_value)
        l2_loss_value = l2_loss_value.mean() 
    
        ssim_loss_value = self.ssim_loss(predicts, labels).mean()
        
        aux_loss_value = 0
        if (len(mid_results)!=0) and (mid_results is not None):
            for i in range(len(mid_results)):
                b, c, h, w = mid_results[i].shape
                gt_downsample = F.interpolate(labels, size=(h, w), mode="bilinear", align_corners=True)
                aux_loss_value += self.l2_loss_aux(mid_results[i], gt_downsample)
                aux_loss_value += self.ssim_loss_aux(mid_results[i], gt_downsample).mean()
            aux_loss_value /= len(mid_results)
            
        # compute overall loss
        l2_loss_value = l2_loss_value * self.args.l2_loss_weight
        ssim_loss_value = ssim_loss_value * self.args.ssim_loss_weight
        aux_loss_value = aux_loss_value * self.args.aux_loss_weight
        
        loss_all = l2_loss_value + ssim_loss_value + aux_loss_value

        loss_dict = {
            'loss_all': loss_all, 
            'loss_mse': l2_loss_value,
            'loss_ssim': ssim_loss_value,
            'loss_aux': aux_loss_value,
        }
        
        return loss_dict


if __name__ == '__main__':
    from config import args
    x = torch.rand(4,3,128,128)
    y = torch.rand(4,3,128,128)
    loss = RawVideoLoss(args)
    
    loss(x, [], y)