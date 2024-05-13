#coding=utf-8
import cv2
from pyzbar.pyzbar import decode
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssdlite320_mobilenet_v3_large
import os


class BarCodeDet:
    # pt模型保存位置
    os.environ['TORCH_HOME'] = 'F:/pythonProject/Product_Review_Analysis/models/'

    def __init__(self, image_path):
        self.barCode = None
        self.image = None
        # 加载已经训练好的模型
        model = ssdlite320_mobilenet_v3_large(weights='SSDLite320_MobileNet_V3_Large_Weights.DEFAULT')
        model.eval()

        # 图像预处理函数
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # 读取图像
        self.image = cv2.imread(image_path)

        # 将图像转换为PyTorch张量并添加批次维度
        input_tensor = transform(self.image).unsqueeze(0)

        # 使用模型进行目标检测
        with torch.no_grad():
            prediction = model(input_tensor)

        # 解析检测结果并绘制边界框
        for score, label, bbox in zip(prediction[0]['scores'], prediction[0]['labels'], prediction[0]['boxes']):
            if score > 0.5 and label == 1:  # 如果置信度大于0.5且标签为1（表示检测到的是物体）
                x, y, w, h = map(int, bbox)
                cv2.rectangle(self.image, (x, y), (w, h), (0, 255, 0), 2)

        # 使用ZBar库来识别条形码
        barCodeRectangle = []
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        for barcode in decode(gray_image):
            data = barcode.data.decode('utf-8')
            x, y, w, h = barcode.rect
            barCodeRectangle.append(barcode.rect)
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(self.image, data, (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6)
            self.barCode = data

        if self.barCode is not None:
            print("条形码：" + self.barCode)
        else:
            print("未找到条形码！请让照片更清晰或者条形码更明显！")

    # 显示图片
    def show_dispicture(self):
        # 显示图片
        height, width, _ = self.image.shape
        # 创建一个窗口
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        # 调整窗口大小以适应图像
        cv2.resizeWindow('Image', width, height)
        cv2.imshow('Image', self.image)
        cv2.waitKey(0)  # 等待按下任意键
        cv2.destroyAllWindows()

    # 构造参数解析并分析参数
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True, help="path to the image file")
    # args = vars(ap.parse_args())
    # image = cv2.imread(args["image"])

    # image_path = "data/images/weib-6972434756270.jpg"
    # image_path = "data/images/yida-6923450656181.jpg"
