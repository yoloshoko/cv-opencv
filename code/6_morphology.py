import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from utils.data_utils import *



if __name__ == "__main__":
    img = cv2.imread("../data/images/dige.png")
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

    
    res = np.vstack((img,dige_erosion,dige_dilate,opening,closing))
    res = cv2.resize(res,(187, 660))
    cv_show(res)
    