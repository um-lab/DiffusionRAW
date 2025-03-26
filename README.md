
<div align="center">
  <h1>Leveraging Frame Affinity for sRGB-to-RAW Video De-rendering   (CVPR 2024)</h1>


[Chen Zhang](https://scholar.google.com/citations?user=qRcKyw0AAAAJ&hl=zh-CN)<sup>1,\*</sup>
| [Wencheng Han](https://scholar.google.com/citations?user=hGZueIUAAAAJ&hl=zh-CN&oi=ao)<sup>2,\*</sup>
| Yang Zhou<sup>1</sup>
| [Jianbing Shen](https://scholar.google.com/citations?user=_Q3NTToAAAAJ&hl=zh-CN&oi=ao)<sup>2,†</sup>
| [Cheng-zhong Xu](https://scholar.google.com/citations?user=XsBBTUgAAAAJ&hl=zh-CN&oi=ao)<sup>2</sup>
| [Wentao Liu](https://scholar.google.com/citations?user=KZn9NWEAAAAJ&hl=zh-CN&oi=ao)<sup>1,†</sup>

<sup>1</sup> *SenseTime Research and Tetras.AI*, 
<sup>2</sup> *SKL-IOTSC, CIS, University of Macau*

<sup>*</sup> Equal Contribution. <sup>†</sup> Corresponding Authors.
</div>

# 🔔 Overview
### Hightlights
- We propose a new architecture for **RAW video derendering**. This architecture can efficiently de-render
RAW video sequences using **only one RAW frame** and
sRGB videos as input. By adopting this method, both
storage and computation efficiency for RAW video capturing can be significantly improved.

<p align="center">
<img src="./assets/video_raw_motivation.jpg" alt="示例图片" width="400">
</p>

- We propose **a new benchmark** for RAW video derendering to comprehensively evaluate the methods for
this task. To our knowledge, this is the **first benchmark**
specifically designed for the RAW video de-rendering
task.
<p align="center">
<img src="./assets/dataset.png" alt="dataset" width="400">
</p>

### Framework
The framework consists of two main stages:
1. **Temporal Affinity Prior Extraction:**
This stage generates a reference RAW image by leveraging motion information between adjacent frames.
2. **Spatial Feature Fusion and Mapping:**
Using the reference RAW as the initial state, a pixel-level mapping function is learned to refine inaccurately predicted pixels from the first stage. This process incorporates guidance from the sRGB image and preceding frames.

<p align="center">
<img src="./assets/framework.png" alt="dataset" width="800">
</p>

# ⏰ TODO List

- [ ] Dataset Release
- [x] Model Release
- [x] Code Release


# 🔧 Installation
Our model does not use any hard-to-configure packages. You only need to install torch and some simple dependencies (such as numpy, cv2). Of course, you can directly follow the steps below to configure the environment:
```
# git clone this repository
git clone https://github.com/zhangchen98/RAW_CVPR24.git
cd RAW_CVPR24

# create an environment
conda create -n videoRaw python=3.8
conda activate videoRaw
pip install -r requirements.txt
```

# 📁 Dataset


# 🔥Training

**Train the model on the RVD-Part1 dataset:**
```
python3 -u main.py \
--trainset_root='./RVD/Part1/train' \
--testset_root='./RVD/Part1/test' \
--input_size="900,1600" \
--save_dir='./checkpoints/RVD_Part1' \
--batch_size=2 \
--test_freq=20 \
--patch_size=256 \
--load_from='' \
--port=12355 \
--max_epoch=60 \
--num_worker=8 \
--init_lr=0.002 \
--lr_decay_epoch=20 \
--aux_loss_weight=0.5 \
--ssim_loss_weight=1.0 \
--local
```
**Train the model on the RVD-Part2 dataset:**
```
python3 -u main.py \
--trainset_root='./RVD/Part2/train' \
--testset_root='./RVD/Part2/test' \
--input_size="640,1440" \
--save_dir='./checkpoints/RVD_Part2' \
--batch_size=2 \
--test_freq=20 \
--patch_size=256 \
--load_from='' \
--port=12347 \
--max_epoch=60 \
--num_worker=8 \
--init_lr=0.002 \
--lr_decay_epoch=20 \
--aux_loss_weight=0.5 \
--ssim_loss_weight=1.0 \
--local \
```
You can also amend the startup script in 'scripts' folder to use multi-GPU training. 

# ⚡ Checkpoints and Inference
1. Download the pretrained models (RVD_Part1.pth, RVD_Part2.pth) from [BaiduYun](https://pan.baidu.com/s/1wBTyalAq_k-nBXpMsWuBYQ) (code: axh6).
2. Put the pretrained models in the './pretrain' folder.

3. Run the test script:
```
# test on RVD-Part1
python3 -u main.py \
--trainset_root='./RVD/Part1/train' \
--testset_root='./RVD/Part1/test' \
--input_size="900,1600" \
--save_dir='./checkpoints/RVD_Part1' \
--batch_size=8 \
--test_freq=20 \
--patch_size=256 \
--load_from='./pretrain/RVD_Part1.pth' \
--port=12355 \
--max_epoch=60 \
--num_worker=8 \
--init_lr=0.002 \
--lr_decay_epoch=20 \
--aux_loss_weight=0.5 \
--ssim_loss_weight=1.0 \
--local \
--test_only \
# --save_predict_raw  # add this option to save the predicted raw images
```
```
# test on RVD-Part2
python3 -u main.py \
--trainset_root='./RVD/Part2/train' \
--testset_root='./RVD/Part2/test' \
--input_size="640,1440" \
--save_dir='./checkpoints/RVD_Part2' \
--batch_size=8 \
--test_freq=20 \
--patch_size=256 \
--load_from='./pretrain/RVD_Part2.pth' \
--port=12347 \
--max_epoch=60 \
--num_worker=8 \
--init_lr=0.002 \
--lr_decay_epoch=20 \
--aux_loss_weight=0.5 \
--ssim_loss_weight=1.0 \
--local \
--test_only \
# --save_predict_raw # add this option to save the predicted raw images
```
You can find the testing results in the `./checkpoints/RVD_Part1` and `./checkpoints/RVD_Part2` directories.

# 🥰 Acknowledgements
Our dataset contains part of the data from Real-RawVSR Dataset(https://github.com/zmzhang1998/Real-RawVSR), thanks to the excellent work of Yue et al.

# 🎓 Citation
```
@inproceedings{zhang2024leveraging,
  title={Leveraging Frame Affinity for sRGB-to-RAW Video De-rendering},
  author={Zhang, Chen and Han, Wencheng and Zhou, Yang and Shen, Jianbing and Xu, Cheng-zhong and Liu, Wentao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={25659--25668},
  year={2024}
}
```
# ✉️ Contact
If you have any questions, please contact: zhangchen2@tetras.ai