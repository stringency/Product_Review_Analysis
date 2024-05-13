#coding=utf-8
from barcode_det import BarCodeDet
from barcode_rec import BarCodeRec

# 按装订区域中的绿色按钮以运行脚本。
# if __name__ == '__main__':
# image_path = "data/images/weib-6972434756270.jpg"
# 检测图片中的条形码
image_path = "data/images/yida-6923450656181.jpg"
barCodeDetector = BarCodeDet(image_path)
# 展示识别到的条形码
barCodeDetector.show_dispicture()
# 识别获取到的条形码，并且提取商品信息
barCodeDetector = BarCodeRec(barCodeDetector.barCode)
json_info = barCodeDetector.requestT1()
dict_info = json_info['json']
print(dict_info['code_name']+" "+dict_info['code_spec'])
