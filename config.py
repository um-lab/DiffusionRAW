import argparse


parser = argparse.ArgumentParser(description='Video Raw Reconstruction')

# Dataset
parser.add_argument('--trainset_root', default=None, type=str, help='root dir of train data')
parser.add_argument('--trainset_json', default=None, type=str, help='path of train json')
parser.add_argument('--testset_root', default=None, type=str, help='root dir of test data')
parser.add_argument('--testset_json', default=None, type=str, help='path of test json')
parser.add_argument('--patch_size', default=256, type=int, help='patch size for training')
parser.add_argument('--raw_bit_depth', default=14, type=int, help='bit depth of raw image')
parser.add_argument('--aug_ratio', default=0.2, type=float, help='the ratio of data augmentation')
parser.add_argument('--input_size', default='640,1440', type=str, help='h,w of input image') 

# Training
parser.add_argument('--init_lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--lr_decay_epoch', default=20, type=int, help='epoch of learning rate adjustment')
parser.add_argument('--lr_decay_gamma', default=0.1, type=float, help='factor of learning rate adjustment')
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--batch_size', default=8, type=int, help='batch size for training')
parser.add_argument('--num_worker', default=8, type=int, help='num_workers for dataloader')
parser.add_argument('--max_epoch', default=60, type=int, help='max_epoches for training')
parser.add_argument('--start_epoch', default=0, type=int, help='start_epoch for training')
parser.add_argument('--save_freq', default=5, type=int, help='save interval (epoches)')
parser.add_argument('--test_freq', default=20, type=int, help='test interval (epoches)')
parser.add_argument('--warmup_epoch', default=0, type=int, help='how many epoches for warmup')
parser.add_argument('--warmup_factor', default=0.1, type=float, help='factor for warmup')
parser.add_argument('--save_dir', default='./checkpoints/default', type=str, help='path to save checkpoints')
parser.add_argument('--load_from', default=None, type=str, help='checkpoint for pretrain or test')
parser.add_argument('--resume_from', default=None, type=str, help='checkpoint for resume')
parser.add_argument('--seed', default=24, type=int, help="randmon seed")
parser.add_argument('--port', default=12345, type=int, metavar='P', help='master port')
parser.add_argument('--local', default=False, action='store_true', help='train on local machine or not')

# Testing
parser.add_argument('--test_only', action='store_true', help='test mode')
parser.add_argument('--save_predict_raw', action='store_true', help='save results')

# Loss
parser.add_argument('--l2_loss_weight', default=1, type=float, help='the weight of L2 loss')
parser.add_argument('--ssim_loss_weight', default=1, type=float, help='the weight of SSIM loss')
parser.add_argument('--aux_loss_weight', default=0.5, type=float, help='the weight of Auxiliary loss')

args = parser.parse_args()



