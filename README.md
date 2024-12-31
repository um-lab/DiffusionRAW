

<h1 style="text-align: center; font-weight: bold;">Leveraging Frame Affinity for sRGB-to-RAW Video De-rendering</h1>


![PDF as Image](./assets/video_raw_motivation.jpg)


# TODO List

- [x] Dataset Release
- [x] Model Release
- [] Code Release


# Installation
Our model does not use any hard-to-configure packages. You only need to install torch and some simple dependencies (such as numpy, cv2). By the way, we use pytorch 1.3 to implement the proposed method.

# Dataset


# Training
You can amend the startup script in 'scripts' folder to run our method. 

```
sh scripts/RVD_Part1.sh
sh scripts/RVD_Part2.sh
```

# Checkpoints and Inference


# Acknowledgements
Our dataset contains part of the data from Real-RawVSR Dataset(https://github.com/zmzhang1998/Real-RawVSR), thanks to the excellent work of Yue et al.

# Citation
```
@inproceedings{zhang2024leveraging,
  title={Leveraging Frame Affinity for sRGB-to-RAW Video De-rendering},
  author={Zhang, Chen and Han, Wencheng and Zhou, Yang and Shen, Jianbing and Xu, Cheng-zhong and Liu, Wentao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={25659--25668},
  year={2024}
}