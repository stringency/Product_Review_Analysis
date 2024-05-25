import csv
import os

import pandas as pd
import torch
import onnx
import onnxruntime
from torch import nn
from transformers import BertModel, BertTokenizer


# 模型结构
# 网络结构
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out


class BertOnnxPerdict:
    def __init__(self, scriptDirectory="F:\\pythonProject\\Product_Review_Analysis",
                 barCode="6923450656181",
                 BEST_MODEL_PATH="models/bert/finetune/bert_dnn_140.model",
                 MODEL_PATH="models/bert/chinese_wwm_pytorch",
                 path_comments="F:/pythonProject/Product_Review_Analysis/output/6923450656181/comments.csv",
                 path_commentsSentiment="F:\\pythonProject\\Product_Review_Analysis" + "\\output\\6923450656181\\commentsSentiment_onnx.csv",

                 ):

        self.scriptDirectory = scriptDirectory
        self.barCode = barCode

        # 检测设备
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # device = "cpu"
        print(self.device)

        # 加载微调好的PyTorch模型
        self.BEST_MODEL_PATH = BEST_MODEL_PATH
        self.net = torch.load(self.BEST_MODEL_PATH)
        self.net.eval()
        # Out[133]:
        # Net(
        #   (fc): Linear(in_features=768, out_features=1, bias=True)
        #   (sigmoid): Sigmoid()
        # )

        # 创建示例输入
        self.MODEL_PATH = MODEL_PATH
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_PATH)  # 分词器
        self.bert = BertModel.from_pretrained(self.MODEL_PATH)  # 模型

        # 评论获取
        self.path_comments = path_comments
        self.df_comments = pd.read_csv(self.path_comments)
        # 删除包含 NaN 值的行
        self.df_comments.dropna(subset=['comment'], inplace=True)
        # df_comments.head(5)
        # df转列表
        self.ls_comments = self.df_comments["comment"].tolist()
        print(self.ls_comments[:5])

        self.path_commentsSentiment = path_commentsSentiment

    # 导出模型为ONNX格式
    # dummy_input = (input_ids, attention_mask)
    def mkdir(self, path):
        # 去除首位空格
        path = path.strip()
        # 去除尾部 \ 符号
        path = path.rstrip("\\")
        # 判断路径是否存在
        isExists = os.path.exists(path)
        # 判断结果
        if not isExists:
            os.makedirs(path)
            print(path + ' 创建成功')
            return True
        else:
            # 如果目录存在则不创建，并提示目录已存在
            print(path + ' 目录已存在')
            return False

    # 保存模型预测结果
    # save_to_csv(products, os.path.join(self.scriptDirectory, "output", self.barCode, 'products.csv'),fieldnames=['url', 'title'])
    def save_to_csv(self, products, path, fieldnames):
        """将商品数据保存到 CSV 文件"""
        # 判断路径是否存在
        isExists = os.path.exists(path)

        with open(path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # 如果文件不存在，则写入表头
            if not isExists:
                writer.writeheader()
            # 写入商品数据
            writer.writerows(products)

    # 开始模型预测
    def bert_onnx_predicted(self):
        print(self.scriptDirectory)
        # 创建文件夹
        self.mkdir(self.scriptDirectory + '\\' + "output" + '\\' + self.barCode)
        # 保存评论预测的文件路径
        # commentsSentiment_path = os.path.join(scriptDirectory, "output", barCode, 'commentsSentiment.csv')
        # 判断路径是否存在
        commentsSentiment_isExists = os.path.exists(self.path_commentsSentiment)
        # 删除文件
        if commentsSentiment_isExists:
            os.remove(self.path_commentsSentiment)

        self.mkdir("models/bert/onnx")
        # 评论预测
        for s in self.ls_comments:
            if len(s) > 500:
                s = s[:500]
            tokens = self.tokenizer([s], padding=True)
            input_ids = torch.tensor(tokens["input_ids"]).to(self.device)
            attention_mask = torch.tensor(tokens["attention_mask"]).to(self.device)
            self.bert.to(self.device)
            # print("input_ids device:", input_ids.device)
            # print("attention_mask device:", attention_mask.device)
            # print("bert device:", bert.device)
            last_hidden_states = self.bert(input_ids, attention_mask=attention_mask)
            bert_output = last_hidden_states[0][:, 0]

            output_path = "models/bert/onnx/bert_dnn.onnx"
            torch.onnx.export(self.net, bert_output, output_path, opset_version=11)

            # 加载ONNX模型并进行预测
            onnx_model = onnx.load(output_path)
            onnx_session = onnxruntime.InferenceSession(output_path)
            input_data = {
                'onnx::Gemm_0': bert_output.cpu().detach().numpy()
            }

            onnx_outputs = onnx_session.run(None, input_data)
            # print(onnx_outputs)

            outputs = self.net(bert_output)
            # print(outputs)
            # emotion_labels = ["积极情感" if item > 0.5 else "消极情感" for item in outputs]
            # for sentence, emotion in zip([s], emotion_labels):
            #     print(sentence,"->", emotion)
            for label_emotion, sentence in zip([1 if item > 0.5 else 0 for item in outputs], [s]):
                # print(label_emotion, sentence)
                self.save_to_csv([{'label_emotion': label_emotion, 'sentence': sentence}],
                                 self.path_commentsSentiment,
                                 fieldnames=['label_emotion', 'sentence'])

# scriptDirectory = "F:\\pythonProject\\Product_Review_Analysis"
# barCode = "6923450656181"
#
# bert_onnx_pre = BertOnnxPerdict(
#     scriptDirectory=scriptDirectory,
#     barCode=barCode,
#     MODEL_PATH="./models/bert/chinese_wwm_pytorch",  # 预训练模型
#     BEST_MODEL_PATH="./models/bert/finetune/bert_dnn_140.model",  # 微调最佳的模型
#     path_comments="./output/" + barCode + "/comments.csv",  # 评论文件位置
#     path_commentsSentiment=scriptDirectory + "\\output\\" + barCode + "\\commentsSentiment_onnx.csv"
# )
#
# bert_onnx_pre.bert_onnx_predicted()
