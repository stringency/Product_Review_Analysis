import numpy as np
import onnx
import onnxruntime

from utils import load_corpus_bert
import pandas as pd
import os
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics

TRAIN_PATH = "./data/comsentiment/weibo2018/train.txt"
TEST_PATH = "./data/comsentiment/weibo2018/test.txt"

# 分别加载训练集和测试集
train_data = load_corpus_bert(TRAIN_PATH)
test_data = load_corpus_bert(TEST_PATH)

df_train = pd.DataFrame(train_data, columns=["text", "label"])
df_test = pd.DataFrame(test_data, columns=["text", "label"])
df_train.head()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 在我的电脑上不加这一句, bert模型会报错
MODEL_PATH = "F:\\pythonProject\\Product_Review_Analysis\\models\\bert\\chinese_wwm_pytorch"

# 加载
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)  # 分词器
bert = BertModel.from_pretrained(MODEL_PATH)  # 模型

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(device)

# 超参数
learning_rate = 1e-3
input_size = 768
num_epoches = 10
batch_size = 100
decay_rate = 0.9


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


# net = Net(input_size).to(device)
# 获取微调后的模型
BEST_MODEL_PATH = "./models/bert/finetune/bert_dnn_140.model"
net = torch.load(BEST_MODEL_PATH).to(device)  # 训练过程中的巅峰时刻


# 数据集
class MyDataset(Dataset):
    def __init__(self, df):
        self.data = df["text"].tolist()
        self.label = df["label"].tolist()

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)


# 训练集
train_data = MyDataset(df_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 测试集
test_data = MyDataset(df_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# 测试集效果检验
def test():
    y_pred, y_true = [], []

    with torch.no_grad():
        for words, labels in test_loader:
            tokens = tokenizer(words, padding=True)
            input_ids = torch.tensor(tokens["input_ids"]).to(device)
            attention_mask = torch.tensor(tokens["attention_mask"]).to(device)
            bert.to(device)
            last_hidden_states = bert(input_ids, attention_mask=attention_mask)
            bert_output = last_hidden_states[0][:, 0]
            outputs = net(bert_output)  # 前向传播
            outputs = outputs.view(-1)  # 将输出展平
            y_pred.append(outputs)
            y_true.append(labels)

    y_prob = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    y_pred = y_prob.clone()
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    print(metrics.classification_report(y_true.cpu(), y_pred.cpu()))
    print("准确率:", metrics.accuracy_score(y_true.cpu(), y_pred.cpu()))
    print("AUC:", metrics.roc_auc_score(y_true.cpu(), y_prob.cpu()))


def onnx_test():
    y_pred, y_true = [], []

    with torch.no_grad():
        for words, labels in test_loader:
            tokens = tokenizer(words, padding=True)
            input_ids = torch.tensor(tokens["input_ids"]).to(device)
            attention_mask = torch.tensor(tokens["attention_mask"]).to(device)
            bert.to(device)
            last_hidden_states = bert(input_ids, attention_mask=attention_mask)
            bert_output = last_hidden_states[0][:, 0]

            output_path = "models/bert/onnx/bert_dnn.onnx"
            torch.onnx.export(net, bert_output, output_path, opset_version=11)
            # 加载ONNX模型并进行预测
            onnx_model = onnx.load(output_path)
            onnx_session = onnxruntime.InferenceSession(output_path)
            input_data = {
                'onnx::Gemm_0': bert_output.cpu().detach().numpy()
            }
            onnx_outputs = onnx_session.run(None, input_data)
            # print(onnx_outputs)
            # 转为torch.Tensor
            onnx_outputs = torch.Tensor(np.array(onnx_outputs)).to(device)

            # outputs = net(bert_output)  # 前向传播
            outputs = onnx_outputs.view(-1)  # 将输出展平
            y_pred.append(outputs)
            y_true.append(labels)

    y_prob = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    y_pred = y_prob.clone()
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    print(metrics.classification_report(y_true.cpu(), y_pred.cpu()))
    print("准确率:", metrics.accuracy_score(y_true.cpu(), y_pred.cpu()))
    print("AUC:", metrics.roc_auc_score(y_true.cpu(), y_prob.cpu()))


"""
下面两个是测试函数
"""
# test()
# onnx_test()
