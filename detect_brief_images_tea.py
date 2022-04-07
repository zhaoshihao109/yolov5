from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import scale_coords,non_max_suppression
from utils.plots import Annotator,colors
import numpy as np
import torch
import cv2
import threading
from copy import deepcopy
import os
import tqdm

"""
处理照片每张照片显示百分比
"""
# 摄像头or视频
camera_id = 0
img_height = 1080
img_width = 1920
thread_lock = threading.Lock()
thread_exit = False
saveVideo = False
class myThread(threading.Thread):
    def __init__(self, camera_id, img_height, img_width):
        super(myThread, self).__init__()
        self.camera_id = camera_id
        self.img_height = img_height
        self.img_width = img_width
        self.frame = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(3,img_width) #设置分辨率
        self.cap.set(4,img_height)
    def get_frame(self):
        return deepcopy(self.frame)

    def run(self):
        global thread_exit

        while not thread_exit:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (self.img_width, self.img_height))
                thread_lock.acquire()
                self.frame = frame
                thread_lock.release()
            else:
                thread_exit = True
        self.cap.release()


# yolo检测设置
conf_thres = 0.35
iou_thres = 0.45
line_thickness=2  # bounding box thickness (pixels)

# 加载模型
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
# model_weight_path = "yolov5x6.pt"
model_weight_path = "./runs/train/exp3/weights/teav1.2.pt"
model = DetectMultiBackend(model_weight_path, device=device, dnn=False)
if device.type != 'cpu':
    model(torch.zeros(1, 3, 640, 640).to(device).type_as(next(model.model.parameters())))  # warmup

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
            return img0

    return img0

if __name__ == '__main__':
    # get_Video()
    originFileRoot = r'F:\postgraduate study\Project\tea_lavel\demo\IMG_2520.JPG'
    dstFileRoot = r'F:\postgraduate study\Project\tea_lavel\demo\result3.jpg'
    img0 = cv2.imread(originFileRoot)
    img0 = caulateFrame(img0)
    cv2.imwrite(dstFileRoot, img0)