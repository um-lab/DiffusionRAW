#!/usr/bin/env bash
set -x

PARTITION=Test  
JOB_NAME=RVD_Part1
GPUS=1
GPUS_PER_NODE=${GPUS:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
TRAINSET_ROOT='/mnt/lustrenew/share/zhangchen2/Dataset/Canon5D3/train'
TESTSER_ROOT='/mnt/lustrenew/share/zhangchen2/Dataset/Canon5D3/test'
SAVE_PATH='./checkpoints/RVD_Part1'
PRETRAIN_MODEL='./pretrain/RVD_Part1.pth'

mkdir -p $SAVE_PATH

export PYTORCH_VERSION=1.3

srun --mpi=pmi2 -p ${PARTITION} --job-name=${JOB_NAME} --gres=gpu:${GPUS_PER_NODE} -n$GPUS --quotatype=auto \
    --ntasks-per-node=${GPUS_PER_NODE} --cpus-per-task=${CPUS_PER_TASK} --kill-on-bad-exit=1 \
    python3 -u main.py \
    --trainset_root=${TRAINSET_ROOT} \
    --testset_root=${TESTSER_ROOT} \
    --input_size="900,1600" \
    --save_dir=${SAVE_PATH} \
    --batch_size=8 \
    --test_freq=20 \
    --patch_size=256 \
    --load_from=${PRETRAIN_MODEL} \
    --port=12355 \
    --max_epoch=60 \
    --num_worker=8 \
    --init_lr=0.002 \
    --lr_decay_epoch=20 \
    --aux_loss_weight=0.5 \
    --ssim_loss_weight=1.0 \
    --test_only
