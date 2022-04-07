from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import scale_coords,non_max_suppression
from utils.plots import Annotator,colors
import torch
import requests
from io import BytesIO
import numpy as np
from PIL import Image
import cv2
import sys

"""
处理照片每张照片显示百分比
当做java 的脚本
"""

# yolo检测设置
conf_thres = 0.35
iou_thres = 0.45
line_thickness=2  # bounding box thickness (pixels)

# 加载模型
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
# model_weight_path = "yolov5x6.pt"
model_weight_path = "./runs/train/exp5/weights/teav1.4.pt"
model = DetectMultiBackend(model_weight_path, device=device, dnn=False)
if device.type != 'cpu':
    model(torch.zeros(1, 3, 640, 640).to(device).type_as(next(model.model.parameters())))  # warmup

class urlImage:
    def __init__(self,url=None):
        if url == None:
            print("please input url")
            return
        self.img = None
        self.url = url
    def __getImage(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            response_byte = response.content
            bytes_stream = BytesIO(response_byte)
            capture_img = Image.open(bytes_stream)
            capture_img = cv2.cvtColor(np.asarray(capture_img), cv2.COLOR_RGB2BGR)
            self.img = capture_img
            return True,self.img
        else:
            return False,None
    def showImage(self):
        if self.img == None:
            ret,img = self.__getImage()
            if(ret==True):
                self.img = img
            else:
                print('url error')
                return None
        # cv2.imshow("capture_img", self.img)
        # cv2.waitKey(0)
        return self.img

# 处理照片 检测
def caulateFrame(img0):
    img = letterbox(img0, [640, 640], stride=64, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255
    img = img[None]  # expand for batch dim

    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)
    names = model.names

    for det in pred:
        if len(det):
            annotator = Annotator(img0, line_width=line_thickness, example=str(names))
            imgwh = img.shape[2:]
            img0shape = img0.shape
            detbbox = det[:, :4]

            det[:, :4] = scale_coords(imgwh, detbbox, img0shape).round()
            caulateNumberOfclassDict = {'1':0,'2':0,'3':0,'4':0,'5':0}
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                caulateNumberOfclassDict[names[c]] += 1
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
            # Stream results
            img0 = annotator.result()
            numberTea1 = caulateNumberOfclassDict['1']
            numberTea2 = caulateNumberOfclassDict['2']
            numberTea3 = caulateNumberOfclassDict['3']
            numberTea4 = caulateNumberOfclassDict['4']
            numberTea5 = caulateNumberOfclassDict['5']
            teaAll = numberTea1 + numberTea2 + numberTea3 + numberTea4 + numberTea5
            putStr = "tea1:" + str(int(numberTea1/teaAll *100)) + "% " + "tea2:" + str(int(numberTea2/teaAll *100)) + "% "+ "tea3:" + str(int(numberTea3/teaAll *100)) + "% "+ "tea4:" + str(int(numberTea4/teaAll *100)) + "% "+ "tea5:" + str(int(numberTea5/teaAll *100)) + "% "

            img0 = cv2.putText(img0, putStr, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255, ), 2)
            # cv2.imshow("asd",img0)
            # cv2.waitKey(0)
            return img0,(str(int(numberTea1/teaAll *100)),str(int(numberTea2/teaAll *100)),str(int(numberTea3/teaAll *100)),str(int(numberTea4/teaAll *100)),str(int(numberTea5/teaAll *100))),(numberTea1,numberTea2,numberTea3,numberTea4,numberTea5)
    return img0

if __name__ == '__main__':

    img0 = cv2.imread('testImg.jpg')
    img0 ,teaResultPer,numBer= caulateFrame(img0)
    print("percent",*teaResultPer,"number:",*numBer)
    cv2.imwrite("teaImage1result.jpg",img0)