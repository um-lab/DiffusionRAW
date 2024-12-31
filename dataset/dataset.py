import os
import os.path as path
from PIL import Image
import numpy as np
import json
import random
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def pad_to_multiple_of(image, multiple):
    height, width = image.size(1), image.size(2)
    new_height = int((height + multiple - 1) // multiple) * multiple
    new_width = int((width + multiple - 1) // multiple) * multiple
    padding_top = (new_height - height) // 2
    padding_bottom = new_height - height - padding_top
    padding_left = (new_width - width) // 2
    padding_right = new_width - width - padding_left
    padded_image = F.pad(image, (padding_left, padding_right, padding_top, padding_bottom))
    return padded_image


def raw_1ch_to_4ch(raw_1ch):
    r = raw_1ch[::2, ::2]
    b = raw_1ch[1::2, 1::2]
    gr = raw_1ch[::2, 1::2]
    gb = raw_1ch[1::2, ::2]

    raw4ch = np.stack((r, gr, gb, b), axis=2).astype(np.float32)

    return raw4ch


class RawOneVideosTestDataset(Dataset):

    def __init__(self, root, data, raw_bit_depth, transform=None):
        self.root = root
        self.data = data
        self.raw_bit_depth = raw_bit_depth
        
        self.raw_dir = path.join(self.root, 'RAW')
        self.rgb_dir = path.join(self.root, 'sRGB')
        self.flow_dir = path.join(self.root, 'flow')

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # Add more transformations if needed
            ])
        else:
            self.transform = transform

    def to_tensor(self, img, max_value):
        img = torch.from_numpy(img).permute(2, 0, 1)
        img /= (max_value)
        
        return img
    
    def __len__(self):
        return len(self.data["frames"]) - 1

    def __getitem__(self, idx):
        filename = self.data["frames"][idx]
        next_filename = self.data["frames"][idx + 1]
        
        raw_filename = os.path.join(self.raw_dir, self.data["video_name"],filename + ".tiff")
        next_raw_filename = os.path.join(self.raw_dir, self.data["video_name"], next_filename + ".tiff")
        rgb_filename = os.path.join(self.rgb_dir, self.data["video_name"], filename + ".png")
        next_rgb_filename = os.path.join(self.rgb_dir, self.data["video_name"], next_filename + ".png")
        flow_filename = os.path.join(self.flow_dir, self.data["video_name"], filename + "_flow.npy")
        
        # read img
        raw_image = cv2.imread(raw_filename, cv2.IMREAD_UNCHANGED)
        next_raw_image = cv2.imread(next_raw_filename, cv2.IMREAD_UNCHANGED)    
        rgb_image = Image.open(rgb_filename)
        rgb_image = np.array(rgb_image, dtype=np.float32)
        next_rgb_image = Image.open(next_rgb_filename)
        next_rgb_image = np.array(next_rgb_image, dtype=np.float32)
        flow_data = np.load(flow_filename)
        flow_img = Image.open(flow_filename.replace('.npy', '.png'))
        flow_img = np.array(flow_img, dtype=np.float32)
        
        # normalize flow
        h, w, c = flow_data.shape
        flow_data[:, :, 0] /= w
        flow_data[:, :, 1] /= h
        
        # align size
        raw_image = raw_1ch_to_4ch(raw_image)
        next_raw_image = raw_1ch_to_4ch(next_raw_image)
        rgb_image = cv2.resize(rgb_image, (w // 2, h // 2))
        next_rgb_image = cv2.resize(next_rgb_image, (w // 2, h // 2))
        flow_img = cv2.resize(flow_img, (w // 2, h // 2))
        flow_data = cv2.resize(flow_data, (w // 2, h // 2))

        h, w, c = flow_data.shape
        flow_data[:, :, 0] *= w
        flow_data[:, :, 1] *= h
        
        # np.array to tensor
        raw_image = self.to_tensor(raw_image, 2**self.raw_bit_depth-1)
        next_raw_image = self.to_tensor(next_raw_image, 2**self.raw_bit_depth-1)
        rgb_image = self.to_tensor(rgb_image, 255)
        next_rgb_image = self.to_tensor(next_rgb_image, 255)
        flow_data = self.to_tensor(flow_data, 1)
        flow_img = self.to_tensor(flow_img, 255)

        # padding
        raw_image = pad_to_multiple_of(raw_image, 32).to(torch.float32)
        next_raw_image = pad_to_multiple_of(next_raw_image, 32).to(torch.float32)
        rgb_image = pad_to_multiple_of(rgb_image, 32).to(torch.float32)
        next_rgb_image = pad_to_multiple_of(next_rgb_image, 32).to(torch.float32)
        flow_data = pad_to_multiple_of(flow_data, 32).to(torch.float32)
        flow_img = pad_to_multiple_of(flow_img, 32).to(torch.float32)

        return raw_image, next_raw_image, rgb_image, next_rgb_image, flow_data, flow_img, next_raw_filename


class RawVideosTestDataset(Dataset):

    def __init__(self, json_file):
        self.data = json.load(open(json_file, 'r'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class RawVideoDataset(Dataset):
    
    def __init__(self, root, json_file, patch_size, raw_bit_depth, aug_ratio, transform=None):
        self.root = root
        self.data = json.load(open(json_file, 'r'))
        self.raw_dir = path.join(self.root, 'RAW')
        self.rgb_dir = path.join(self.root, 'sRGB')
        self.flow_dir = path.join(self.root, 'flow')
        self.patch_size = int(patch_size)
        self.raw_bit_depth = int(raw_bit_depth)
        self.aug_ratio = aug_ratio
        
        self.pairs = []
        for video in self.data:
            frames = video['frames']
            for i in range(len(frames) - 1):
                from_frame = path.join(video["video_name"], frames[i])
                to_frame = path.join(video["video_name"], frames[i + 1])
                self.pairs.append((from_frame, to_frame))

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # Add more transformations if needed
            ])
        
    def random_crop(self, image_list, patch_size):
        H, W, _ = image_list[0].shape
        rnd_h = random.randint(0, max(0, H - patch_size))
        rnd_w = random.randint(0, max(0, W - patch_size))
        for i in range(len(image_list)):
            image_list[i] = image_list[i][rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
        return image_list
    
    def random_rotate(self, image_list):
        idx = random.randint(0, 3)
        image_list = [np.rot90(img, k=idx).copy() for img in image_list]
        flow_mask = np.ones_like((image_list[-1]))
        if idx == 1:
            flow_mask[:, :, 1] *= -1
            image_list[-1] = image_list[-1][:,:,::-1] * flow_mask
        elif idx == 2:
            flow_mask *= -1 
            image_list[-1] *= flow_mask
        elif idx == 3:
            flow_mask[:, :, 0] *= -1
            image_list[-1] = image_list[-1][:,:,::-1] * flow_mask

        return image_list

    def random_flip(self, image_list):
        idx = random.randint(0, 1)
        image_list = [np.flip(img, axis=idx).copy() for img in image_list]
        if idx == 0:
            image_list[-1][:, :, 1] *= -1
        elif idx == 1:
            image_list[-1][:, :, 0] *= -1
            
        return image_list

    def aug(self, raw_image, rgb_image, next_raw_image, next_rgb_image, flow_img, flow_data):
        img_list = [raw_image, rgb_image, next_raw_image, next_rgb_image, flow_img, flow_data]
        img_list = self.random_crop(img_list, self.patch_size)
        if np.random.rand() < self.aug_ratio:
            img_list = self.random_rotate(img_list)
            img_list = self.random_flip(img_list)
        [raw_image, rgb_image, next_raw_image, next_rgb_image, flow_img, flow_data] = img_list

        return raw_image, rgb_image, next_raw_image, next_rgb_image, flow_img, flow_data
    
    def to_tensor(self, img, max_value):
        img = torch.from_numpy(img).permute(2, 0, 1)
        img /= (max_value)
        
        return img
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        from_frame, to_frame = self.pairs[idx]

        raw_filename = os.path.join(self.raw_dir, from_frame + ".tiff")
        next_raw_filename = os.path.join(self.raw_dir, to_frame + ".tiff")
        rgb_filename = os.path.join(self.rgb_dir, from_frame + ".png")
        next_rgb_filename = os.path.join(self.rgb_dir, to_frame + ".png")
        flow_filename = os.path.join(self.flow_dir, from_frame + "_flow.npy")

        # read img and convert to np.array
        raw_image = cv2.imread(raw_filename, cv2.IMREAD_UNCHANGED)
        next_raw_image = cv2.imread(next_raw_filename, cv2.IMREAD_UNCHANGED)      
        rgb_image = Image.open(rgb_filename)
        rgb_image = np.array(rgb_image, dtype=np.float32)
        next_rgb_image = Image.open(next_rgb_filename)
        next_rgb_image = np.array(next_rgb_image, dtype=np.float32)
        flow_img = Image.open(flow_filename.replace('.npy', '.png'))
        flow_img = np.array(flow_img, dtype=np.float32)
        flow_data = np.load(flow_filename) 
        # normalize flow
        h, w, _ = flow_data.shape
        flow_data[:, :, 0] /= w
        flow_data[:, :, 1] /= h
        # align size
        raw_image = raw_1ch_to_4ch(raw_image)
        next_raw_image = raw_1ch_to_4ch(next_raw_image)
        rgb_image = cv2.resize(rgb_image, (w // 2, h // 2))
        next_rgb_image = cv2.resize(next_rgb_image, (w // 2, h // 2))
        flow_img = cv2.resize(flow_img, (w // 2, h // 2))
        flow_data = cv2.resize(flow_data, (w // 2, h // 2))
        
        h, w, _ = flow_data.shape
        flow_data[:, :, 0] *= w
        flow_data[:, :, 1] *= h
        # augmentation
        raw_image, rgb_image, next_raw_image, next_rgb_image, flow_img, flow_data = \
            self.aug(raw_image, rgb_image, next_raw_image, next_rgb_image, flow_img, flow_data)
        # np.array to tensor
        raw_image = self.to_tensor(raw_image, 2**self.raw_bit_depth-1)
        next_raw_image = self.to_tensor(next_raw_image, 2**self.raw_bit_depth-1)
        rgb_image = self.to_tensor(rgb_image, 255)
        next_rgb_image = self.to_tensor(next_rgb_image, 255)
        flow_data = self.to_tensor(flow_data, 1)
        flow_img = self.to_tensor(flow_img, 255)
        
        # padding
        raw_image = pad_to_multiple_of(raw_image, 32).to(torch.float32)
        next_raw_image = pad_to_multiple_of(next_raw_image, 32).to(torch.float32)
        rgb_image = pad_to_multiple_of(rgb_image, 32).to(torch.float32)
        next_rgb_image = pad_to_multiple_of(next_rgb_image, 32).to(torch.float32)
        flow_data = pad_to_multiple_of(flow_data, 32).to(torch.float32)
        flow_img = pad_to_multiple_of(flow_img, 32).to(torch.float32)

        # print(raw_image.max(), raw_image.min(), next_rgb_image.max(), next_rgb_image.min())
        # print(raw_image.shape, next_raw_image.shape, rgb_image.shape, next_rgb_image.shape, flow_data.shape, flow_img.shape)
        return raw_image, next_raw_image, rgb_image, next_rgb_image, flow_data, flow_img


def build_dataloader(args):
    if args.trainset_json is None:
        args.trainset_json = os.path.join(args.trainset_root, 'data.json')
    dataset = RawVideoDataset(root=args.trainset_root, 
                              json_file=args.trainset_json,
                              patch_size=args.patch_size,
                              raw_bit_depth=args.raw_bit_depth,
                              aug_ratio=args.aug_ratio,
                              )
    train_sampler = DistributedSampler(dataset)
    train_loader = DataLoader(dataset, 
                              batch_size=args.batch_size, 
                              shuffle=(train_sampler is None), 
                              num_workers=args.num_worker, 
                              sampler=train_sampler, 
                              pin_memory=True,
                              drop_last=True)

    if args.testset_json is None:
        args.testset_json = os.path.join(args.testset_root, 'data.json')
    testset = RawVideosTestDataset(json_file=args.testset_json)

    return train_loader, train_sampler, testset



if __name__ == '__main__':
    dataset = RawVideoDataset(root="/mnt/lustrenew/share/zhangchen2/Dataset/VideoRawData/", 
                              json_file="/mnt/lustrenew/share/zhangchen2/Dataset/VideoRawData/data.json",
                              patch_size=512,
                              raw_bit_depth=14,
                              aug_ratio=0.2,
                              )
    a = dataset[0]
