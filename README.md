# 商品评论分析

## 项目简介

1. 通过商品的条形码获取商品信息
2. 通过商品信息在商品销售平台爬取商品评论
3. 通过对商品评论进行情感分析，得到情感分析结果

## 项目目录分析
```
一下文件夹可能没有或者信息不完整，需要自己构建或者自动生成的文件夹
data/
models/
output/
venv/
__pycache__/
```
> tree.exe -L 2
```
.
|-- README.md           ### 使用教程
|-- README_RX.md
|-- __pycache__
|   |-- barcode_det.cpython-310.pyc
|   |-- barcode_rec.cpython-310.pyc
|   |-- barcode_rec2.cpython-310.pyc
|   |-- bert_onnx_predict.cpython-310.pyc
|   |-- bert_predict.cpython-310.pyc
|   |-- bert_train.cpython-310.pyc
|   |-- comments_analysis.cpython-310.pyc
|   |-- test_model.cpython-310.pyc
|   `-- utils.cpython-310.pyc
|-- barcode_det.py
|-- barcode_rec.py
|-- barcode_rec2.py
|-- bert_onnx_predict.py
|-- bert_predict.py
|-- bert_train.py
|-- comments_analysis.py
|-- crawlinfo
|   `-- tb
|-- data
|   |-- analysis           ### 数据分析需要的停用词、词云图
|   |-- backups            ### 自己备份的数据可以存在这里
|   |-- comsentiment          ### bert微调数据集
|   |-- images          ### 商品条形码图片
|   `-- userinfo
|-- main.py          ### 主函数，一键启动
|-- models           ### 模型文件夹
|   |-- bert
|   `-- hub
|-- msg_logger
|   |-- __pycache__
|   `-- coderec_logger.py
|-- msg_seckill.log
|-- output           ### 所有的输出结果,注意备份
|   |-- 6901294179165
|   |-- 6920999705042
|   `-- 6923450656181
|-- reference.md           ### 本项目的参考文献
|-- requirements.txt
|-- test.py          ### 测试模块
|-- test_model.py          ### 模型测试模块
|-- utils.py            ### bert情感分析所需要的工具类
|-- venv
|   |-- Lib
|   |-- Library
|   |-- Scripts
|   |-- etc
|   |-- pyvenv.cfg
|   `-- share
`-- verification_code.png
```


## 环境说明<a id="Environmental_Statement"></a>

1. python>=3.10.4
2. 可以使用GPU或者CPU，安装部分不同的python依赖包
   - torch==2.3.0 torchaudio==2.3.0 torchvision==0.18.0
3. windows11系统

## 启动项目

1. git init
2. git clone 项目地址
3. cd 项目根目录(挺关键的)
4. 创建一个虚拟环境，可以用pycharm打开这个项目，然后会提示给你创建一个虚拟环境
   - 如果实在不会，可以手动安装
   - ```pip install virtualenv```
   - 如果你只有一个python版本，直接创建虚拟环境；否则需要加上python版本的参数
   - ```virtualenv venv```
   - 激活环境(由于windows和linux激活方法不同，这里不细说，就是进去venv找到activate.bat或者activate文件执行)
   - 最后一步最好 cd 回到项目根目录，良好习惯！
5. cpu则需要手动安装属于cpu的依赖包，替换对应的gpu依赖(详细参考[环境说明](#Environmental_Statement))
   - ```pip install -r requirements.txt```
6. 完善data文件夹，自己导入商品条形码
7. 完善models文件夹，自己导入bert模型等
8. output文件夹(大概率)可以不用管，会自动生成
9. 修改crawlinfo/tb/cominfo.py的login_taobao()函数，填入自己的淘宝账号密码
10. 仔细查看main中的传入参数和超参数，自己按照需求修改，然后运行main.py(集成了全部功能)，可以根据需求只运行部分功能
11. 根目录的各个功能模块能够支持单独运行，这样就能提供灵活的调试选择，部分模块也能通过main.py超参数进行控制调试。
12. 注意output内的文件可能会因为二次运行而先删除再生成，所以应当对需要数据进行备份，以防数据丢失。
11. 更多问题，如：如何完善文件夹、main.py内容看不懂、根目录下其他文件看不懂等，可以查看reference.md的参考文献
 