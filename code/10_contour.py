import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from utils.data_utils import *


### 轮廓的性质
# 可以获取全部的轮廓;
# 并获取每个轮廓的周长/面积
# 同时可以获取每个轮廓的多边形的近似/边界矩形/外接圆


if __name__ == "__main__":

    # 采用黑白图进行绘制轮廓会更加准确
    img = cv2.imread("../data/images/contours.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 彩色→灰度 [单通道，0-255 连续值],即灰度图
    _ ,out = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  #灰度→二值 [单通道,只有0和255],即黑白图


    #* 获取轮廓
    # return 二值图像/轮廓/层级关系
    binary, contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # H,W->H,W /<class 'tuple'>/1,N,4
    print(out.shape,"->",binary.shape, type(contours), hierarchy.shape)
    for i, contour in enumerate(contours):
        print(f"contour {i}: {contour.shape}")  # (N, 1, 2):N个(x,y)
    res = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 2)  # -1表示画所有轮廓;B,G,R=(0,0,255);粗细为2单位
    # cv_show(res,'res')
    

    #* 轮廓特征
    cnt = contours[5]
    res = cv2.drawContours(img.copy(), [cnt], -1, (0, 0, 255), 2)
    print("S=",cv2.contourArea(cnt),"\nC=",cv2.arcLength(cnt,True))  #周长，True表示闭合的
    # cv_show(res,'res')


    #* 轮廓近似
    epsilon = 0.005*cv2.arcLength(cnt,True) 
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    print(cnt.shape,'->',approx.shape)  # L,1,2 -> l,1,2
    res = cv2.drawContours(img.copy(), [approx], -1, (0, 0, 255), 2)
    # cv_show(res,'res')


    #* 边界矩形或外接圆
    x,y,w,h = cv2.boundingRect(cnt)
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    # cv_show(img,'img')

    (x,y),radius = cv2.minEnclosingCircle(cnt) 
    center = (int(x),int(y));radius = int(radius) 
    img = cv2.circle(img,center,radius,(0,255,0),2)
    # cv_show(img,'img')