# 可微渲染报告——康乐园

**孔令宇，李宇轩，于沛楠**

## 简介

本项目包含了第二届计图挑战赛计图 - 可微渲染新视角生成比赛的代码实现。本项目的特点是：对于不同的应用使用了不同的可微渲染算法。对于 Coffee 和 Scar 两个应用，我们选择 JNeRF 算法，对于 Car 我们使用了 Jrender 算法，对于 Easyship 我们使用了 Mip-NeRF 算法。对于 Scarf 我们使用了 JNeRF+RL 的方法。

## 安装

本项目可以在一张 3090 显卡上运行

### 运行环境：

- ubuntu 20.04 LTS
- python=3.8
- jittor>=1.3.4.13

### 安装依赖：

对于 Jrender 执行以下命令完成环境安装：

```
cd Jrender
pip install -r requirements.txt
```

对于 JNeRF，执行以下命令安装 NeRF 环境依赖：

```
cd JNeRF
cd python
python -m pip install -e .
```

对于 Mip-NeRF，执行以下命令完成环境安装：

```
cd mip-NeRF
pip install -r requirements.txt
```

## 训练：

Jrender 算法训练：

```
python nerf.py --config ./configs/<场景名>.txt
```

JNeRF 算法训练：

```
python tools/run_net.py --config-file ./projects/ngp/configs/ngp_comp.py
```

Mip-NeRF 算法训练：

```
python train.py --scene <场景名>
```
