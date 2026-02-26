import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from utils.data_utils import *

## 图像阈值处理

if __name__ == "__main__":
    # img = sim_img()  # H,W,C
    img_gray = cv2.imread("../data/images/lena.jpg",cv2.IMREAD_GRAYSCALE)  # H,W,C->H,W
    
    #      0/x/max     ↓     0/x/max
    # -----------------------------------> x轴
    thr1, out1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  # 0,max
    _, out2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)  # max,0
    _, out3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)  # x,max  [注意:没有max,x的情况]
    _, out4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)  # 0,x
    _, out5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)  # x,0
    print(img_gray.shape,"->",thr1,'/',out1.shape)  # H,W- > thr/H,W

    titles = ['Gray Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img_gray, out1, out2, out3, out4, out5]
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    