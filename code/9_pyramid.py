import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from utils.data_utils import *

## 下采样(↓)是变小;上采样(↑)是放大;
## 拉普拉斯金字塔的目的是提取图片的轮廓信息

### 请手搓一个上采样和下采样的代码 !important

if __name__ == "__main__":
    img = cv2.imread("../data/images/lena.jpg")  # H,W,C
    h,w,c = img.shape
    img = img[:h-h%2, :w-w%2]


    #* 上采样和下采样
    up=cv2.pyrUp(img)  # H,W,C->2H,2W,C
    down=cv2.pyrDown(img)  # H,W,C->H/2,W/2,C
    print(img.shape,'->',up.shape,down.shape)
    # cv_show(img);cv_show(up);cv_show(down)


    #* 拉普拉斯金字塔
    # 提取图片的轮廓信息
    down=cv2.pyrDown(img)
    down_up=cv2.pyrUp(down)
    l_1=img-down_up
    # cv_show(l_1,'l_1')


    ## 练习
    up=pyUp(img)
    down=pyrDown(img)
    # cv_show(img);cv_show(up);cv_show(down)