from utils import load_corpus_bert
import pandas as pd
import os
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics


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


class BertFinetune:
    def __init__(self, scriptDirectory="F:\\pythonProject\\Product_Review_Analysis",
                 TRAIN_PATH="./data/comsentiment/weibo2018/train.txt",
                 TEST_PATH="./data/comsentiment/weibo2018/test.txt",
                 MODEL_PATH="F:\\pythonProject\\Product_Review_Analysis\\models\\bert\\chinese_wwm_pytorch",
                 # 超参数
                 learning_rate=1e-3,
                 input_size=768,
                 num_epoches=10,
                 batch_size=100,
                 decay_rate=0.9,
                 # 模型保存路径
                 save_model_path="F:\\pythonProject\\Product_Review_Analysis" + "\\models\\bert\\finetune",
                 ):

        self.scriptDirectory = scriptDirectory
        self.TRAIN_PATH = TRAIN_PATH
        self.TEST_PATH = TEST_PATH

        # 分别加载训练集和测试集
        self.train_data = load_corpus_bert(TRAIN_PATH)
        self.test_data = load_corpus_bert(TEST_PATH)

        self.df_train = pd.DataFrame(self.train_data, columns=["text", "label"])
        self.df_test = pd.DataFrame(self.test_data, columns=["text", "label"])
        self.df_train.head()

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 在我的电脑上不加这一句, bert模型会报错
        # MODEL_PATH = "./model/bert/chinese_wwm_pytorch"  # 下载地址见 https://github.com/ymcui/Chinese-BERT-wwm
        self.MODEL_PATH = MODEL_PATH  # 下载地址见 https://github.com/ymcui/Chinese-BERT-wwm

        # 加载
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)  # 分词器
        self.bert = BertModel.from_pretrained(MODEL_PATH)  # 模型

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # device = "cpu"
        print(self.device)

        # 超参数
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.num_epoches = num_epoches
        self.batch_size = batch_size
        self.decay_rate = decay_rate

        # 训练集
        self.train_data = self.MyDataset(self.df_train)
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)

        # 测试集
        self.test_data = self.MyDataset(self.df_test)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=True)

        self.net = Net(input_size).to(self.device)

        # 定义损失函数和优化器
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=decay_rate)

        # 模型保存路径
        self.save_model_path = save_model_path

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

    # 测试集效果检验
    def test(self):
        y_pred, y_true = [], []

        with torch.no_grad():
            for words, labels in self.test_loader:
                tokens = self.tokenizer(words, padding=True)
                input_ids = torch.tensor(tokens["input_ids"]).to(self.device)
                attention_mask = torch.tensor(tokens["attention_mask"]).to(self.device)
                last_hidden_states = self.bert(input_ids, attention_mask=attention_mask)
                bert_output = last_hidden_states[0][:, 0]
                outputs = self.net(bert_output)  # 前向传播
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

    def bert_finetuned(self):
        # 迭代训练
        for epoch in range(self.num_epoches):
            total_loss = 0
            for i, (words, labels) in enumerate(self.train_loader):
                tokens = self.tokenizer(words, padding=True)
                input_ids = torch.tensor(tokens["input_ids"]).to(self.device)
                attention_mask = torch.tensor(tokens["attention_mask"]).to(self.device)
                labels = labels.float().to(self.device)
                self.bert.to(self.device)
                with torch.no_grad():
                    last_hidden_states = self.bert(input_ids, attention_mask=attention_mask)
                    bert_output = last_hidden_states[0][:, 0]
                self.optimizer.zero_grad()  # 梯度清零
                outputs = self.net(bert_output)  # 前向传播
                logits = outputs.view(-1)  # 将输出展平
                loss = self.criterion(logits, labels)  # loss计算
                total_loss += loss
                loss.backward()  # 反向传播，计算梯度
                self.optimizer.step()  # 梯度更新
                if (i + 1) % 10 == 0:
                    print("epoch:{}, step:{}, loss:{}".format(epoch + 1, i + 1, total_loss / 10))
                    total_loss = 0

            # learning_rate decay
            self.scheduler.step()

            # test
            self.test()

            # save model
            self.mkdir(self.save_model_path)
            model_path = self.save_model_path + "\\bert_dnn_{}.model".format(epoch + 1)
            torch.save(self.net, model_path)
            print("saved model: ", model_path)


# scriptDirectory = "F:\\pythonProject\\Product_Review_Analysis"
# bert_tra = BertFinetune(
#     scriptDirectory=scriptDirectory,
#     TRAIN_PATH="./data/comsentiment/weibo2018/train.txt",
#     TEST_PATH="./data/comsentiment/weibo2018/test.txt",
#     MODEL_PATH=scriptDirectory + "\\models\\bert\\chinese_wwm_pytorch",
#     # 超参数
#     learning_rate=1e-3,
#     input_size=768,
#     num_epoches=140,  # 大约七小时微调
#     batch_size=130,  # 三分钟一轮
#     decay_rate=0.9,
#     # 模型保存路径
#     save_model_path="F:\\pythonProject\\Product_Review_Analysis" + "\\models\\bert\\finetune",
# )
# bert_tra.bert_finetuned()
