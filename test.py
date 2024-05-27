import torch

from barcode_det import BarCodeDet
from barcode_rec import BarCodeRec
from bert_onnx_predict import BertOnnxPerdict
from bert_predict import BertPredict
from bert_train import BertFinetune
from comments_analysis import commentsAnalysis
from crawlinfo.tb.cominfo import ComInfo

import sys
from PyQt5 import QtGui
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QWidget, \
    QGridLayout, QLabel, QHBoxLayout
from PyQt5.QtCore import pyqtSignal, QObject, QUrl


# 创建一个自定义信号类，用于在标准输出中显示信息
class Communicate(QObject):
    textWritten = pyqtSignal(str)


# 创建主窗口类
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Print Output Example")  # 设置窗口标题
        self.setGeometry(100, 100, 1000, 600)  # 设置窗口尺寸和位置

        layout = QGridLayout()  # 使用网格布局

        self.input_project_path = QLineEdit("F:\\pythonProject\\Product_Review_Analysis")
        self.input_project_path.setPlaceholderText("F:\\pythonProject\\Product_Review_Analysis")  # 设置输入框的提示内容
        layout.addWidget(self.input_project_path)  # 将输入框放置在第二行第一列

        self.input_image_path = QLineEdit("data/images/liushen-6901294179165.jpg")
        self.input_image_path.setPlaceholderText("data/images/liushen-6901294179165.jpg")  # 设置输入框的提示内容
        layout.addWidget(self.input_image_path)  # 将输入框放置在第二行第一列

        self.input_craw_page = QLineEdit("5")
        self.input_craw_page.setPlaceholderText("爬取页数")  # 设置输入框的提示内容
        layout.addWidget(self.input_craw_page)  # 将输入框放置在第二行第一列

        self.input_time = QLineEdit("3")
        self.input_time.setPlaceholderText("爬取间隔时间（s）")  # 设置输入框的提示内容
        layout.addWidget(self.input_time)  # 将输入框放置在第二行第一列

        self.input_craw_is = QLineEdit("2")
        self.input_craw_is.setPlaceholderText(
            "爬取信息选项：0:爬取商品信息+评论  1:已有商品信息，只爬取评论  2:不进行数据爬取")  # 设置输入框的提示内容
        layout.addWidget(self.input_craw_is)  # 将输入框放置在第二行第一列

        self.input_bert_is = QLineEdit("0")
        self.input_bert_is.setPlaceholderText("模型是否进行训练：1:进行训练  0:不进行训练只检测")  # 设置输入框的提示内容
        layout.addWidget(self.input_bert_is)  # 将输入框放置在第二行第一列

        self.text_edit = QTextEdit()
        layout.addWidget(self.text_edit)  # 将文本编辑框放置在第一行，占据1列

        self.button_analysis = QPushButton("开始分析")
        self.button_analysis.clicked.connect(self.analysis_start)
        layout.addWidget(self.button_analysis, 10, 0)  # 将按钮放置在第二行第二列

        self.button = QPushButton("查看结果")
        self.button.clicked.connect(self.watch_input)
        layout.addWidget(self.button, 10, 1)  # 将按钮放置在第二行第二列

        # 将布局设置为主窗口的中央组件
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # 创建自定义信号对象
        self.communicate = Communicate()
        # 将自定义信号的文本传递信号连接到显示输出的方法
        self.communicate.textWritten.connect(self.normal_output_written)

        # 重定向标准输出流到自定义信号对象
        sys.stdout = EmittingStream(self.communicate)

        # self.scriptDirectory = "F:\\pythonProject\\Product_Review_Analysis"
        # self.image_path = "data/images/liushen-6901294179165.jpg"
        # self.max_page = 5
        # self.index_page_time = 3
        # self.option_fun = 2  # 0:爬取商品信息+评论  1:已有商品信息，只爬取评论  2:不进行数据爬取
        # self.is_train = 0
        self.barCode = None

    # 提交输入的方法
    def analysis_start(self):
        # 项目绝对路径
        scriptDirectory = str(self.input_project_path.text())
        image_path = str(self.input_image_path.text())
        max_page = int(self.input_craw_page.text())
        index_page_time = int(self.input_time.text())  # 由于多处用这个时间，爬取一页商品信息大概5-8秒，大概一个商品爬取评论需要10秒
        option_fun = int(self.input_craw_is.text())  # 0:爬取商品信息+评论  1:已有商品信息，只爬取评论  2:不进行数据爬取
        is_train = int(self.input_bert_is.text())

        print(scriptDirectory)
        print(image_path)
        print(max_page)
        print(index_page_time)
        print(option_fun)
        print(is_train)

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
        # 商品条形码
        self.barCode = barCodeDetector.barCode
        KEYWORD = dict_info['code_name'] + " " + dict_info['code_spec']
        # 爬取的页数
        cur_page = 1

        selenium_tb_tor = ComInfo(barCode=self.barCode, scriptDirectory=scriptDirectory, KEYWORD=KEYWORD, cur_page=cur_page,
                                  max_page=max_page, index_page_time=index_page_time)

        if option_fun != 2:
            selenium_tb_tor.selenium_tb(option_fun)
        print("信息获取结束")

        # # 评论预测
        # # 设备检测
        # device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # print(device)
        # if is_train == 1:
        #     # bert模型微调
        #     bert_tra = BertFinetune(
        #         scriptDirectory=scriptDirectory,
        #         TRAIN_PATH="./data/comsentiment/weibo2018/train.txt",
        #         TEST_PATH="./data/comsentiment/weibo2018/test.txt",
        #         MODEL_PATH=scriptDirectory + "\\models\\bert\\chinese_wwm_pytorch",
        #         # 超参数
        #         learning_rate=1e-3,
        #         input_size=768,
        #         num_epoches=140,  # 大约七小时微调
        #         batch_size=130,  # 三分钟一轮
        #         decay_rate=0.9,
        #         # 模型保存路径
        #         save_model_path="F:\\pythonProject\\Product_Review_Analysis" + "\\models\\bert\\finetune",
        #     )
        #     bert_tra.bert_finetuned()
        #     print("Bert模型微调结束")
        # else:
        #     # 对于有的模型进行测试
        #
        #     from torch import nn
        #
        #     class Net(nn.Module):
        #         def __init__(self, input_size):
        #             super(Net, self).__init__()
        #             self.fc = nn.Linear(input_size, 1)
        #             self.sigmoid = nn.Sigmoid()
        #
        #         def forward(self, x):
        #             out = self.fc(x)
        #             out = self.sigmoid(out)
        #             return out
        #
        #     import test_model
        #
        #     test_model.test()
        #     test_model.onnx_test()
        #     print("Bert模型测试结束，Bert_onnx模型测试结束")
        # # bert模型预测
        # bert_pre = BertPredict(
        #     scriptDirectory=scriptDirectory,
        #     barCode=barCode,
        #     MODEL_PATH="./models/bert/chinese_wwm_pytorch",  # 预训练模型
        #     BEST_MODEL_PATH="./models/bert/finetune/bert_dnn_140.model",  # 微调最佳的模型
        #     path_comments="./output/" + barCode + "/comments.csv"  # 评论文件位置
        # )
        # bert_pre.bert_predicted()
        # print("Bert模型预测结束")
        # # bert转onnx模型预测，并进行预测
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
        # print("Bert模型转onnx模型预测结束")
        # # 评论数据分析
        # commentsAnalysis(scriptDirectory=scriptDirectory,
        #                  barCode=barCode,
        #                  path_stopwords=scriptDirectory + "\\data\\analysis\\stopwords.txt",
        #                  # 已经分好类的评论
        #                  path_commentsSentiment_onnx=scriptDirectory + "\\output\\" + barCode + "\\commentsSentiment_onnx.csv",
        #                  # 生成文件夹,用于存储数据分析的结果文件
        #                  path_save_commentsAnalysis=scriptDirectory + '\\' + "output" + '\\' + barCode + '\\' + "commentsAnalysis")
        #
        # print("评论预测结束")

    def watch_input(self):
        folder_path = "./output/"+self.barCode
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path))
        self.parameter = self.input_time.text()  # 获取输入框中的文本作为参数
        # self.input_edit.clear()  # 清空输入框
        print(f"Received parameter: {folder_path}")  # 打印接收到的参数到标准输出

    # 显示输出到文本编辑框的方法
    def normal_output_written(self, text):
        cursor = self.text_edit.textCursor()  # 获取文本编辑框的光标
        cursor.movePosition(QtGui.QTextCursor.End)  # 将光标移动到文本的末尾
        cursor.insertText(text)  # 在光标位置插入文本
        self.text_edit.setTextCursor(cursor)  # 设置文本编辑框的光标
        self.text_edit.ensureCursorVisible()  # 确保光标可见


# 创建用于重定向标准输出流的类
class EmittingStream:
    def __init__(self, communicate):
        self.communicate = communicate

    # 重写 write 方法，将输出文本传递给自定义信号对象
    def write(self, text):
        self.communicate.textWritten.emit(str(text))


if __name__ == "__main__":
    app = QApplication(sys.argv)  # 创建应用程序对象
    window = MainWindow()  # 创建主窗口对象
    window.show()  # 显示主窗口
    print("123")  # 在标准输出中打印信息
    sys.exit(app.exec_())  # 运行应用程序并进入事件循环
