from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import scale_coords,non_max_suppression
from utils.plots import Annotator,colors
import numpy as np
import torch
import cv2
from lxml import etree
import os

"""
加载茶叶图片生成xml标签
"""
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

class XmlMaker:
    def __init__(self,xmlSaveDirName,objInfList,cls,size):
        self.xmlFileName = xmlSaveDirName
        self.objInfList = objInfList
        self.cls = cls
        self.size = size
        self.root = None

    def makexml(self):
        self.root = etree.Element("annotation")

        folder = etree.SubElement(self.root, "folder")
        folder.text = self.cls

        filename = etree.SubElement(self.root, "filename")
        filename.text = os.path.basename(self.xmlFileName)

        path = etree.SubElement(self.root, "path")
        path.text = self.xmlFileName

        source = etree.SubElement(self.root, "source")
        database = etree.SubElement(source,"database")
        database.text = "Unknown"

        size = etree.SubElement(self.root, "size")
        width = etree.SubElement(size, "width")
        width.text = str(self.size[1])
        height = etree.SubElement(size, "height")
        height.text = str(self.size[0])
        depth = etree.SubElement(size, "depth")
        depth.text = str(self.size[2])

        segmented = etree.SubElement(self.root, "segmented")
        segmented.text = "0"
        for objInf in self.objInfList:
            self.makeObject(objInf)
        tree = etree.ElementTree(self.root)
        tree.write(self.xmlFileName, pretty_print=True, xml_declaration=False, encoding='utf-8')
    def makeObject(self,objInf):
        object = etree.SubElement(self.root, "object")
        name = etree.SubElement(object,"name")
        name.text = self.cls
        pose = etree.SubElement(object,"pose")
        pose.text = "Unspecified"
        truncated = etree.SubElement(object,"truncated")
        truncated.text = "0"
        difficult = etree.SubElement(object,"difficult")
        difficult.text = "0"

        bndbox = etree.SubElement(object, "bndbox")
        xmin =  etree.SubElement(bndbox, "xmin")
        xmin.text = str(objInf[1][0])
        ymin = etree.SubElement(bndbox, "ymin")
        ymin.text = str(objInf[1][1])
        xmax = etree.SubElement(bndbox, "xmax")
        xmax.text = str(objInf[1][2])
        ymax = etree.SubElement(bndbox, "ymax")
        ymax.text = str(objInf[1][3])




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
            objList = []
            annotator = Annotator(img0, line_width=line_thickness, example=str(names))
            imgwh = img.shape[2:]
            img0shape = img0.shape
            detbbox = det[:, :4]

            det[:, :4] = scale_coords(imgwh, detbbox, img0shape).round()
            caulateNumberOfclassDict = {'1':0,'2':0,'3':0,'4':0,'5':0}
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{names[c]}'
                objList.append((label,(int(xyxy[0].data),int(xyxy[1].data),int(xyxy[2].data),int(xyxy[3].data))))


    return objList

def makeOneXml(originFileRoot,cls):
    # originFileRoot = r'F:\postgraduate study\Project\tea_lavel\3_26_self_shoot\1\IMG_2512.JPG'
    dstFileRoot = originFileRoot[0:-3]+"xml"
    img0 = cv2.imread(originFileRoot,cv2.IMREAD_COLOR+cv2.IMREAD_IGNORE_ORIENTATION)
    objList = caulateFrame(img0)
    xmlMaker = XmlMaker(dstFileRoot,objList,cls,img0.shape)
    xmlMaker.makexml()
if __name__ == '__main__':
    # get_Video()
    rootDir = r"F:\postgraduate study\Project\tea_lavel\TEA_SINGLE_OVERWIEW_5_v0.4\111"
    jpgFileList = os.listdir(rootDir)
    for jpgFile in jpgFileList:
        makeOneXml(os.path.join(rootDir,jpgFile),"1")

    # cv2.imwrite(dstFileRoot, img0)