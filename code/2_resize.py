import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from utils.data_utils import *



if __name__ == "__main__":
    #* H,W大小变化
    img = cv2.imread("../data/images/lena.jpg")
    img1 = img[0:200,0:200,:]
    print(img.shape,"->",img1.shape)  # H,W,C -> H_1,W_1,C
    # cv_show(img1)

    
    '''
    def resize(src, dsize, fx=None, fy=None):
    if dsize != (0, 0):
        # 使用dsize
        target_w, target_h = dsize
    else:
        # 使用fx,fy计算
        target_w = int(src.shape[1] * fx)
        target_h = int(src.shape[0] * fy)
    ''' 
    # (0,0)类似于-1，让计算机自动计算尺寸
    img2 = cv2.resize(img, (0,0),fy=3,fx=2)  # H,W,C -> 3H,2W,C 
    print(img.shape,"->",img2.shape)
    img3 = cv2.resize(img, (600,800))  # H,W,C -> H_1,W_1,C
    print(img.shape,"->",img3.shape)
    # cv_show(img2)



    #* 各维度的C提取
    b,g,r=cv2.split(img)  # H,W,C->H,W 
    print(img.shape,"->",b.shape) 
    img_merge=cv2.merge((b,g,r))  # H,W->H,W,C  
    print(b.shape,"->",img_merge.shape)



    #* 只显示单通道
    img_1channel = show_1channel(img)
    # cv_show(img_1channel)