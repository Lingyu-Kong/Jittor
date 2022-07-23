# Jrender 2.0 (Jittor渲染库)

## 使用

使用JRender前需要安装好Jittor，Jittor安装方法在[此处](https://github.com/Jittor/jittor)。

然后调用 `pip install -r requirements.txt` 安装以下包：

```
jittor
imageio==2.9.0
imageio-ffmpeg==0.4.3
matplotlib==3.3.0
configargparse==1.3
tensorboard==1.14.0
tqdm==4.46.0
opencv-python==4.2.0.34
```

修改 `configs/<场景名>.txt` 中的配置，`datadir` 表示训练数据存放位置，`basedir` 表示新视角图片生成位置，`expname` 表示场景名。

调用 `python nerf.py --config ./configs/<场景名>.txt` 即可运行

## Citation

如果您在自己的研究工作中使用了JRender，请引用Jittor的论文。
```
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--222103},
  year={2020}
}
```

同时，本渲染器内置了N3MR和SoftRas两个可微渲染器，若您在研究中使用了渲染器，请您引用相应的论文。
```
@InProceedings{kato2018renderer
    title={Neural 3D Mesh Renderer},
    author={Kato, Hiroharu and Ushiku, Yoshitaka and Harada, Tatsuya},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2018}
}

@article{liu2019softras,
  title={Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning},
  author={Liu, Shichen and Li, Tianye and Chen, Weikai and Li, Hao},
  journal={The IEEE International Conference on Computer Vision (ICCV)},
  month = {Oct},
  year={2019}
}
```
