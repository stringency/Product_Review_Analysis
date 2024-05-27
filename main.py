# coding=utf-8
"""
@Time: 2024/5/31 16:06
@File: main.py
@Repositories: https://github.com/stringency/Product_Review_Analysis
"""
import torch

from barcode_det import BarCodeDet
from barcode_rec import BarCodeRec
from bert_onnx_predict import BertOnnxPerdict
from bert_predict import BertPredict
from bert_train import BertFinetune
from comments_analysis import commentsAnalysis
from crawlinfo.tb.cominfo import ComInfo

# 按装订区域中的绿色按钮以运行脚本。
# if __name__ == '__main__':
# image_path = "data/images/weib-6972434756270.jpg"
# 检测图片中的条形码
image_path = "data/images/liushen-6901294179165.jpg"
barCodeDetector = BarCodeDet(image_path)
print("识别结束")
# 展示识别到的条形码
# barCodeDetector.show_dispicture()
# 识别获取到的条形码，并且提取商品信息
barCodeRector = BarCodeRec(barCodeDetector.barCode)
# 开始识别并且获取信息
# 开始前先用浏览器打开这个网页：http://tiaoma.cnaidc.com/
json_info = barCodeRector.requestT1()
dict_info = json_info['json']
print(dict_info['code_name'] + " " + dict_info['code_spec'])
print("信息获取结束")
"""
先让Chrome浏览器进入调试状态的终端命令：
chrome.exe --remote-debugging-port=9222 --incognito
查询端口：
netstat -ano|findstr "9222"
"""
"""
益达木糖醇薄荷40粒瓶装56g 瓶装
六神驱蚊花露水95ml 95ml
"""
# 项目绝对路径
scriptDirectory = "F:\\pythonProject\\Product_Review_Analysis"
# 商品条形码
barCode = barCodeDetector.barCode
KEYWORD = dict_info['code_name'] + " " + dict_info['code_spec']
# 爬取的页数
cur_page = 1
max_page = 5
index_page_time = 3  # 由于多处用这个时间，爬取一页商品信息大概5-8秒，大概一个商品爬取评论需要10秒
selenium_tb_tor = ComInfo(barCode=barCode, scriptDirectory=scriptDirectory, KEYWORD=KEYWORD, cur_page=cur_page,
                          max_page=max_page, index_page_time=index_page_time)
option_fun = 2  # 0:爬取商品信息+评论  1:已有商品信息，只爬取评论  2:不进行数据爬取
if option_fun != 2:
    selenium_tb_tor.selenium_tb(option_fun)
print("信息获取结束")

# 评论预测
# 设备检测
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
is_train = 0
if is_train == 1:
    # bert模型微调
    bert_tra = BertFinetune(
        scriptDirectory=scriptDirectory,
        TRAIN_PATH="./data/comsentiment/weibo2018/train.txt",
        TEST_PATH="./data/comsentiment/weibo2018/test.txt",
        MODEL_PATH=scriptDirectory + "\\models\\bert\\chinese_wwm_pytorch",
        # 超参数
        learning_rate=1e-3,
        input_size=768,
        num_epoches=140,  # 大约七小时微调
        batch_size=130,  # 三分钟一轮
        decay_rate=0.9,
        # 模型保存路径
        save_model_path="F:\\pythonProject\\Product_Review_Analysis" + "\\models\\bert\\finetune",
    )
    bert_tra.bert_finetuned()
    print("Bert模型微调结束")
else:
    # 对于有的模型进行测试

    from torch import nn

    class Net(nn.Module):
        def __init__(self, input_size):
            super(Net, self).__init__()
            self.fc = nn.Linear(input_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out = self.fc(x)
            out = self.sigmoid(out)
            return out


    import test_model

    test_model.test()
    test_model.onnx_test()
    print("Bert模型测试结束，Bert_onnx模型测试结束")
# bert模型预测
bert_pre = BertPredict(
    scriptDirectory=scriptDirectory,
    barCode=barCode,
    MODEL_PATH="./models/bert/chinese_wwm_pytorch",  # 预训练模型
    BEST_MODEL_PATH="./models/bert/finetune/bert_dnn_140.model",  # 微调最佳的模型
    path_comments="./output/" + barCode + "/comments.csv"  # 评论文件位置
)
bert_pre.bert_predicted()
print("Bert模型预测结束")
# bert转onnx模型预测，并进行预测
bert_onnx_pre = BertOnnxPerdict(
    scriptDirectory=scriptDirectory,
    barCode=barCode,
    MODEL_PATH="./models/bert/chinese_wwm_pytorch",  # 预训练模型
    BEST_MODEL_PATH="./models/bert/finetune/bert_dnn_140.model",  # 微调最佳的模型
    path_comments="./output/" + barCode + "/comments.csv",  # 评论文件位置
    path_commentsSentiment=scriptDirectory + "\\output\\" + barCode + "\\commentsSentiment_onnx.csv"
)

bert_onnx_pre.bert_onnx_predicted()
print("Bert模型转onnx模型预测结束")
# 评论数据分析
commentsAnalysis(scriptDirectory=scriptDirectory,
                 barCode=barCode,
                 path_stopwords=scriptDirectory + "\\data\\analysis\\stopwords.txt",
                 # 已经分好类的评论
                 path_commentsSentiment_onnx=scriptDirectory + "\\output\\" + barCode + "\\commentsSentiment_onnx.csv",
                 # 生成文件夹,用于存储数据分析的结果文件
                 path_save_commentsAnalysis=scriptDirectory + '\\' + "output" + '\\' + barCode + '\\' + "commentsAnalysis")

print("评论预测结束")
