import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import scale_coords,non_max_suppression
from utils.plots import Annotator,colors
import torch
import cv2
import numpy as np
import tqdm

"""
处理照片显示百分比，并且用每一个类的所有数据绘制成饼图,并且每张照片生成百分比的照片
"""
# yolo检测设置
conf_thres = 0.35
iou_thres = 0.45
line_thickness=2  # bounding box thickness (pixels)

# 加载模型
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
# model_weight_path = "yolov5x6.pt"
model_weight_path = "./runs/train/exp4/weights/teav1.3.pt"
# model_weight_path = "./runs/train/exp2/weights/teav1.1.pt"
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
        caulateNumberOfclassDict = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
        if len(det):
            annotator = Annotator(img0, line_width=line_thickness, example=str(names))
            imgwh = img.shape[2:]
            img0shape = img0.shape
            detbbox = det[:, :4]

            det[:, :4] = scale_coords(imgwh, detbbox, img0shape).round()

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

            img0 = cv2.putText(img0, putStr, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, ), 1)
            # cv2.imshow("asd",img0)
            # cv2.waitKey(0)

    return img0,caulateNumberOfclassDict

if __name__ == '__main__':
    # get_Video()
    originFileRoot = 'F:\postgraduate study\Project\\tea_lavel\FILE'
    dstFileRoot = 'F:\postgraduate study\Project\\tea_lavel\FILE_dst'
    classList = os.listdir(originFileRoot)
    allClsDict = {'1': {}, '2': {}, '3': {}, '4': {}, '5': {}}

    for cls in tqdm.tqdm(classList):
        if cls == 'all':
            originFileRoot = os.path.join(originFileRoot,cls)
            dstFileRoot = os.path.join(dstFileRoot,cls)
            allClassList = os.listdir(originFileRoot)
            for Cls in tqdm.tqdm(allClassList):
                dictClsNumber = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
                originclsRoot = os.path.join(originFileRoot,Cls)
                dstclsRoot = os.path.join(dstFileRoot, Cls)
                if os.path.isdir(dstclsRoot) == False:
                    os.makedirs(dstclsRoot)
                fileNameList = os.listdir(originclsRoot)
                # 为保存文本信息构建路径
                clsRoot = os.path.join(dstFileRoot,Cls)
                saveTxtDir = os.path.join(clsRoot,Cls+".txt")
                with open(saveTxtDir,"w") as file:
                    for fileName in fileNameList:
                        filePath = os.path.join(originclsRoot, fileName)
                        dstPath = os.path.join(dstclsRoot, fileName)
                        img0 = cv2.imread(filePath)
                        img0,dictNumber = caulateFrame(img0)
                        cv2.imwrite(dstPath,img0)
                        file.write(fileName+":")
                        writeStr = str(dictNumber["1"])+ " " + str(dictNumber["2"])+ " " + str(dictNumber["3"]) +  " " + str(dictNumber["4"])+ " " + str(dictNumber["5"])+"\n"
                        file.write(writeStr)
                        dictClsNumber["1"] += dictNumber["1"]
                        dictClsNumber["2"] += dictNumber["2"]
                        dictClsNumber["3"] += dictNumber["3"]
                        dictClsNumber["4"] += dictNumber["4"]
                        dictClsNumber["5"] += dictNumber["5"]
                    numberTea1 = dictClsNumber['1']
                    numberTea2 = dictClsNumber['2']
                    numberTea3 = dictClsNumber['3']
                    numberTea4 = dictClsNumber['4']
                    numberTea5 = dictClsNumber['5']
                    teaAll = numberTea1 + numberTea2 + numberTea3 + numberTea4 + numberTea5
                    dictClsNumber["1"] = int(numberTea1 / teaAll * 100)
                    dictClsNumber["2"] = int(numberTea2 / teaAll * 100)
                    dictClsNumber["3"] = int(numberTea3 / teaAll * 100)
                    dictClsNumber["4"] = int(numberTea4 / teaAll * 100)
                    dictClsNumber["5"] = int(numberTea5 / teaAll * 100)
                    allClsDict[Cls] = dictClsNumber
    name_list = ['1', '2', '3', '4', "5"]
    num_lists = [allClsDict["1"].values(), allClsDict["2"].values(), allClsDict["3"].values(), allClsDict["4"].values(),allClsDict["5"].values(),]
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.bar(range(len(name_list)), num_lists[0], tick_label=name_list)
    plt.subplot(2, 3, 2)
    plt.bar(range(len(name_list)), num_lists[1], tick_label=name_list)
    plt.subplot(2, 3, 3)
    plt.bar(range(len(name_list)), num_lists[2], tick_label=name_list)
    plt.subplot(2, 3, 4)
    plt.bar(range(len(name_list)), num_lists[3], tick_label=name_list)
    plt.subplot(2, 3, 5)
    plt.bar(range(len(name_list)), num_lists[4], tick_label=name_list)
    plt.savefig(os.path.join(dstFileRoot,'teaBar.jpg'))
    plt.show()
