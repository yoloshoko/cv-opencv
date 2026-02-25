import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from utils.data_utils import *



if __name__ == "__main__":
    #* 图片的输入和输出
    # img = sim_img()  # H,W,C
    img = cv2.imread("../data/images/lena.jpg")  # H,W,C
    # cv_show(img)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # H,W,C->H,W
    #
    # <=> 
    #
    img = cv2.imread("../data/images/lena.jpg",cv2.IMREAD_COLOR)  # H,W,C
    img_gray = cv2.imread("../data/images/lena.jpg",cv2.IMREAD_GRAYSCALE)  # H,W,C->H,W
    print(img.shape,"->",img_gray.shape)
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)  # H,W,C->H,W,C
    print(img.shape,"->",img_hsv.shape)

    cv2.imwrite('../data/images/lena_gray.jpg',img_gray)

    print(type(img),img.size,img.dtype)  # ndarray/H*W*C/0-255(xxxxxxxx，8位)



    #* 视频的读取
    vc = cv2.VideoCapture('../data/videos/park.mp4')
    video_show(vc)