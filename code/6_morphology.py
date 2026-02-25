import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from utils.data_utils import *


# 想象一幅图像是一座“地形图”，像素的亮度代表高度（亮的像素高，暗的像素低）。
# 原图：真实的地形，有高山（亮物体），有低谷（暗区域）。
# 开运算：相当于用一个小球（结构元素）在地形表面滚过。小球无法进入的狭窄缝隙（比小球小的亮区域）会被抹平，相当于移除了那些细小的“尖刺”或“窄峰”。
# 顶帽变换（原-开）：拿原来的地形减去抹平了细小尖刺后的地形。剩下的就是那些被抹掉的细小尖峰。这些尖峰就像是戴在头顶上的“高帽子”。

# 腐蚀/开运算结果差不多，形状变小;膨胀/闭运算结果差不多，形状变大;
# 梯度生成轮廓;
# 顶帽提取明噪声，负责抓出那些不该出现的亮点;黑帽提取暗噪声,负责抓出那些不该出现的暗点

if __name__ == "__main__":
    # img = cv2.imread("../data/images/dige.png")
    img = cv2.imread("../data/images/lena_noise.png")
    kernel3 = np.ones((3,3),np.uint8) 
    kernel5 = np.ones((5,5),np.uint8) 
    # cv_show(img,"img")


    #* 腐蚀操作
    dige_erosion = cv2.erode(img,kernel3,iterations = 1)  # H,W,C->H,W,C
    print(dige_erosion.shape)
    # cv_show(dige_dilate,"dige_dilate")


    #* 膨胀操作
    dige_dilate = cv2.dilate(img,kernel3,iterations = 1)  # H,W,C->H,W,C
    print(dige_dilate.shape)
    # cv_show(dige_dilate,"dige_dilate")


    #* 开运算(先腐蚀，再膨胀)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel5)  # H,W,C->H,W,C
    print(opening.shape)
    # cv_show(opening,"opening")


    #* 闭预算(先膨胀，再腐蚀)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel5)  # H,W,C->H,W,C
    print(closing.shape)
    # cv_show(closing,"closing")

    
    #* 形态学梯度运算(膨胀-腐蚀)
    # 生成一个物体的轮廓图 [梯度的定义]
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT,kernel3)  # H,W,C->H,W,C
    print(gradient.shape)
    # cv_show(gradient,"gradient")


    #* 顶帽(原图-开运算)
    # 得到图像中的孤立亮点噪声，用来提取或去除噪声
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel3)  # H,W,C->H,W,C
    print(tophat.shape)
    # cv_show(tophat,"tophat")


    #* 黑帽(闭运算-原图)
    # 提取图像中比背景暗的细小物体或局部区域
    blackhat  = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT, kernel3)  # H,W,C->H,W,C
    print(blackhat.shape)
    # cv_show(blackhat,"blackhat")


    res = np.vstack((img,dige_erosion,dige_dilate,opening,closing,gradient,tophat,blackhat))
    res = cv2.resize(res,(187, 660))
    cv_show(res)
    