import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from utils.data_utils import *

### Canny边缘检测
# (1)使用高斯滤波器,以平滑图像,滤除噪声。
# (2)计算图像中每个像素点的梯度强度和方向。
# (3)应用非极大值(Non-Maximum Suppression)抑制,以消除边缘检测带来的杂散响应。
# (4)应用双阈值(Double-Threshold)检测来确定真实的和潜在的边缘。
# (5)通过抑制孤立的弱边缘最终完成边缘检测。

### 请手搓一个Canny边缘检测的代码 !important

if __name__ == "__main__":
    img = cv2.imread("../data/images/lena.jpg",cv2.IMREAD_GRAYSCALE)  # H,W,C
    # img = cv2.imread("../data/images/lena.jpg")  # H,W
    v1=cv2.Canny(img,80,150)  # H,W,C / H,W -> H,W
    print(img.shape,'->',v1.shape)
    v2=cv2.Canny(img,50,100)  # H,W,C / H,W -> H,W


    ## 练习:
    v3 = canny(img,50,100)
    res = np.hstack((v1,v2,v3))
    cv_show(res,'res')
