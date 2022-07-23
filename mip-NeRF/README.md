# mip-NeRF

### 环境要求

- 在该文件目录下执行pip install -r requirements.txt
- 默认cuda环境已经配置好
- 在data/nerf_synthetic中添加以场景名字命名的文件夹，在文件夹中添加训练数据
  - 训练数据格式为
    - test，文件夹中为验证集中图片
    - train，文件夹中为训练集中图片
    - transforms_test.json，验证集数据
    - transforms_train.json，训练集数据
- 训练命令
  - python train.py --scene <scene_name>
  - 您可以在config.py中观察各参数含义
- 渲染图片
  - 将transforms_test.json中数据修改为您要渲染的图片的数据
  - 在log文件夹中创建以场景名命名的文件夹
  - 在visualize.py中修改category变量为对应场景名
  - 执行python visualize.py即可
