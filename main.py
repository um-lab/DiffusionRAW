import os
import re
import socket
import random
import logging
import json
import numpy as np
import cv2
import time
from thop import profile

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.backends import cudnn
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dataset.dataset import build_dataloader, RawOneVideosTestDataset
from model import RawVideoNetwork
from config import args
from logger import get_logger
from loss import RawVideoLoss
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


device = "cuda" if torch.cuda.is_available() else 'cpu'  
logger = get_logger(args)


def check_keys(model, checkpoint, logger):
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(checkpoint['state_dict'].keys())
    missing_keys = model_keys - ckpt_keys
    for key in missing_keys:
        logger.warning('missing key in model:{}'.format(key))
    unexpected_keys = ckpt_keys - model_keys
    for key in unexpected_keys:
        logger.warning('unexpected key in checkpoint:{}'.format(key))
    # shared_keys = model_keys & ckpt_keys
    # for key in shared_keys:
    #     logger.info('shared key:{}'.format(key))


def main():
    if args.local: 
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        os.environ['MASTER_ADDR'] = 'localhost'
    else:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.local_rank = int(os.environ['SLURM_LOCALID'])
        node_list = str(os.environ['SLURM_NODELIST'])
        node_parts = re.findall('[0-9]+', node_list)
        os.environ['MASTER_ADDR'] = f'{node_parts[1]}.{node_parts[2]}.{node_parts[3]}.{node_parts[4]}'

    os.environ['MASTER_PORT'] = str(args.port)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(args.rank)

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank) 

    if args.rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    logger.info("rank {} of {} jobs, in {}".format(args.rank, args.world_size, socket.gethostname()))
    
    if not args.local:
        dist.barrier()
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # cudnn.deterministic = True    
       
    # Load dataset
    train_loader, train_sampler, testset = build_dataloader(args)
    # Load model
    model = RawVideoNetwork(args)
    model = model.to(device)
    model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
    # logger.info(model)
    
    # Loss and optimize
    criterion = RawVideoLoss(args)
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    schedule = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_epoch, args.lr_decay_gamma)

    if args.resume_from:
        assert os.path.isfile(args.resume_from), f'Not found resume model: {args.resume_from}'
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        # checkpoint['state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        check_keys(model=model, checkpoint=checkpoint, logger=logger)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"resume training from '{args.resume_from}' at epoch {checkpoint['epoch']}")
    elif args.load_from:
        checkpoint = torch.load(args.load_from)
        check_keys(model=model, checkpoint=checkpoint, logger=logger)
        weight = checkpoint['state_dict']
        model.load_state_dict(weight, strict=False)
    
    if args.test_only:
        logger.info("use test_only mode ...")
        if args.rank == 0:
            test(testset, model, args)
        return
    
    # Training loop
    logger.info("======Starting training======")
    for epoch in range(args.start_epoch, args.max_epoch):
        train_sampler.set_epoch(epoch + 1)
        train(train_loader, model, criterion, optimizer, epoch, args)
        if (epoch + 1) % args.test_freq == 0 or epoch == args.max_epoch:
            test(testset, model, args)
        schedule.step()
    logger.info("======Training finished======")


def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    for idx, (raw_image, next_raw_image, rgb_image, next_rgb_image, flow_data, flow_img) in enumerate(train_loader):
        raw_image = raw_image.to(device)
        next_raw_image = next_raw_image.to(device)
        rgb_image = rgb_image.to(device)
        next_rgb_image = next_rgb_image.to(device)
        flow_data = flow_data.to(device)

        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        output, mid_results = model(rgb_image, next_rgb_image, raw_image, flow_data)
        loss_dict = criterion(output, mid_results, next_raw_image)
        loss = loss_dict['loss_all']
        # Backpropagation and optimization
        loss.backward()
        # max_grad_norm = 1.0  # 设置梯度的最大范数
        # clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if idx % 100 == 0:
            logger.info(
                f"Epoch [{epoch+1}/{args.max_epoch}], Iter [{idx}/{len(train_loader)}], "
                f"lr={optimizer.state_dict()['param_groups'][0]['lr']}, "
                f"Loss_all: {loss_dict['loss_all']:.6f}, "
                f"loss_mse: {loss_dict['loss_mse']:.6f}, "
                f"loss_ssim: {loss_dict['loss_ssim']:.6f}, "
                f"loss_aux: {loss_dict['loss_aux']:.6f}"
            )
    
    if args.rank == 0 and (epoch + 1) % args.save_freq == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        checkpoint_filename = os.path.join(args.save_dir, 'model', f"model_{epoch+1}.pth")
        if not os.path.exists(os.path.dirname(checkpoint_filename)):
            os.makedirs(os.path.dirname(checkpoint_filename))
        torch.save(checkpoint, checkpoint_filename)


@torch.no_grad()
def test(testset, model, args):
    model.eval()
    result_dict = {}
    psnr_all_videos, ssim_all_videos = 0, 0
    logger.info("Testing start")
    for video_idx, video in enumerate(testset):
        dataset = RawOneVideosTestDataset(args.testset_root, video, raw_bit_depth=args.raw_bit_depth)
        video_name = video['video_name']
        psnr_one_video, ssim_one_video = 0, 0
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        logger.info(f"Testing dataset - {video_name}")

        for idx, (raw_image, next_raw_image, rgb_image, next_rgb_image, flow_data, flow_img, next_raw_filename) in enumerate(dataloader):
            # load data
            raw_image = raw_image.to(device)
            next_raw_image = next_raw_image.to(device)
            rgb_image = rgb_image.to(device)
            next_rgb_image = next_rgb_image.to(device)
            flow_data = flow_data.to(device)
            psnr_value = 0
            ssim_value = 0
            # predict
            if idx == 0:
                prev_raw = raw_image
            time1 = time.time()
            outputs, others = model(rgb_image, next_rgb_image, prev_raw, flow_data)
            time2 = time.time()
            # print(time2-time1)
            # flops, params = profile(model, inputs=(rgb_image, next_rgb_image, prev_raw, flow_data,))
            # print("FLOPs = " + str(flops/1000**3)+'G')
            # print("Params = " + str(params/1000**2) + 'M')
            input_size = (args.input_size.split(',')[0], args.input_size.split(',')[1])
            input_size = (int(input_size[0])//2, int(input_size[1])//2)
            
            padding_h = (outputs.shape[2] - int(input_size[0])) // 2
            padding_w = (outputs.shape[3] - int(input_size[1])) // 2
            if padding_h !=0:
                outputs[:,:,:padding_h, :] = 0
                outputs[:,:,-padding_h:, :] = 0
            if padding_w != 0:
                outputs[:,:,:, :padding_w] = 0
                outputs[:,:,:, -padding_w:] = 0
            prev_raw = outputs

            # return    
            # compute metric
            pred_raw = outputs.detach()
            gt = next_raw_image.detach()
            if padding_h !=0:
                pred_raw = pred_raw[:,:,padding_h:-padding_h, :]
                gt = gt[:,:,padding_h:-padding_h, :]
            if  padding_w != 0:
                pred_raw = pred_raw[:,:,:, padding_w:-padding_w]
                gt = gt[:,:,:, padding_w:-padding_w]
                        
            pred_raw = pred_raw.squeeze().permute(1,2,0).cpu().numpy()
            gt = gt.squeeze().permute(1,2,0).cpu().numpy()
            
            pred_raw = raw_4ch_to_1ch(pred_raw)
            gt = raw_4ch_to_1ch(gt)
            
            pred_raw = (pred_raw*(2**args.raw_bit_depth-1)).astype(np.uint16)
            gt = (gt*(2**args.raw_bit_depth-1)).astype(np.uint16)
            psnr_value = psnr(pred_raw /(2**args.raw_bit_depth-1), gt/(2**args.raw_bit_depth-1))
            ssim_value = ssim(pred_raw /(2**args.raw_bit_depth-1), gt/(2**args.raw_bit_depth-1))

            if video_name not in result_dict:
                result_dict[video_name] = []
            result_dict[video_name].append(f'psnr: {psnr_value}, ssim: {ssim_value}')
            # save predict
            if args.save_predict_raw:
                save_name = os.path.join(args.save_dir, 'predict/', video_name, next_raw_filename[0].split('/')[-1])
                if not os.path.exists(os.path.dirname(save_name)):
                    os.makedirs(os.path.dirname(save_name))
                cv2.imwrite(save_name, pred_raw)
            psnr_one_video += psnr_value
            ssim_one_video += ssim_value
        psnr_one_video_avg = psnr_one_video / (idx+1)
        ssim_one_video_avg = ssim_one_video / (idx+1)
        logger.info(f"{idx+1} frames, average psnr -- {psnr_one_video_avg:.4f}, average ssim --{ssim_one_video_avg:.4f}")  
        result_dict[video_name].append(f"{idx+1} frames, average psnr -- {psnr_one_video_avg:.4f}, average ssim --{ssim_one_video_avg:.4f}")
        psnr_all_videos += psnr_one_video_avg
        ssim_all_videos += ssim_one_video_avg
    psnr_all_videos_avg = psnr_all_videos / (video_idx+1)
    ssim_all_videos_avg = ssim_all_videos / (video_idx+1)
    logger.info(f"Overall testset, average psnr -- {psnr_all_videos_avg:.4f}, average ssim --{ssim_all_videos_avg:.4f}")
    result_dict["Result"] = f"Overall testset, average psnr -- {psnr_all_videos_avg:.4f}, average ssim --{ssim_all_videos_avg:.4f}"
    
    json.dump(result_dict, open(os.path.join(args.save_dir, "results.json"), "w"), indent=4)


def raw4ch_to_3ch(raw4ch):
    h, w, _ = raw4ch.shape
    raw3ch = np.zeros((h, w, 3), dtype=np.uint8)
    raw3ch[:, :, 0] = raw4ch[:,:,0]
    raw3ch[:, :, 1] = raw4ch[:,:,1]
    raw3ch[:, :, 2] = raw4ch[:,:,2]
    return raw3ch


def raw_4ch_to_1ch(raw_4ch):
    h, w, _ = raw_4ch.shape
    raw_1ch = np.zeros([h*2, w*2])
    
    r = raw_4ch[:, :, 0]
    gr = raw_4ch[:, :, 1]
    gb = raw_4ch[:, :, 2]
    b = raw_4ch[:, :, 3]

    raw_1ch[::2, ::2] = r
    raw_1ch[1::2, 1::2] = b
    raw_1ch[::2, 1::2] = gr
    raw_1ch[1::2, ::2] = gb
    
    return raw_1ch


if __name__ == '__main__':
    main()