# instant_NGP_with_reinforcement

- 环境要求在jnerf的基础上，需要执行
  - pip install baidu-aip
- 第一次运行时，操作与jnerf相同
  - python tools/run_net.py --config-file ./projects/ngp/configs/<对应config文件>
- 数据存放要求与jnerf类似，不同之处在于
  - 您需要将第一次训练生成的测试集图片放在train文件夹中
  - 您还需要将测试集json数据添加在train的json文件数据最后，注意修改对应路径
- 之后的每次训练，您需要
  - 在enhancer.py中修改对应场景名
  - python enhancer.py（注意，这里使用的是百度图像增强api，调用次数有限，超出调用次数会报错）
  - 在alpha_adder.py中修改对应场景名
  - python alpha_adder.py
  - python tools/run_net.py --config-file ./projects/ngp/configs/<对应config文件>
- 之后您需要增强多少次，就从数据存放处要求开始，重复多少次即可
- 更多的信息详见JNeRF文件夹下的README.md