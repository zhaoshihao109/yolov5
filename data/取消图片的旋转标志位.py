import cv2 as cv
import os
root = r"F:\postgraduate study\Project\tea_lavel\TEA_SINGLE_OVERWIEW_5_v0.4\33"
dst = r"F:\postgraduate study\Project\tea_lavel\TEA_SINGLE_OVERWIEW_5_v0.4\333"
# os.mkdir(dst)

for img in os.listdir(root):
    imgPath = os.path.join(root,img)
    imgData = cv.imread(imgPath,cv.IMREAD_COLOR+cv.IMREAD_IGNORE_ORIENTATION)
    cv.imwrite(os.path.join(dst,img),imgData)

