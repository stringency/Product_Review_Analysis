# 商品评论分析

## 项目简介

1. 通过商品的条形码获取商品信息
2. 通过商品信息在商品销售平台爬取商品评论
3. 通过对商品评论进行情感分析，得到情感分析结果

## 项目目录分析
```
需要自己构建/自动生成的文件夹
data/
models/
output/
```
```
|-- README.md
|-- __pycache__
|   |-- barcode_det.cpython-310.pyc
|   `-- barcode_rec.cpython-310.pyc
|-- barcode_det.py
|-- barcode_rec.py
|-- crawlinfo.py
|-- data
|   |-- images
|   `-- userinfo
|-- main.py
|-- models
|   `-- hub
|-- msg_logger
|   |-- __pycache__
|   `-- coderec_logger.py
|-- msg_seckill.log
|-- output
|   `-- 6923450656181
|-- requirements.txt
|-- test.py
|-- venv
|   |-- Lib
|   |-- Library
|   |-- Scripts
|   |-- pyvenv.cfg
|   `-- share
|-- verification_code.png
`-- \312\271\323\303\313\265\303\367.md
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
 