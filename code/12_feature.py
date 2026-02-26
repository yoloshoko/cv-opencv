import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from utils.data_utils import *

### 图像特征检测
# (1)通过Harris进行角点特征检测。找角点的目的是因为角点的特征比较丰富，而平面的特征比较少
# (2)通过SIFT进行关键点特征检测,可以得到Tuple类型的结果,存储多个关键点;还可以得到关键点的特征向量

### 请手搓一个Harris和SIFT进行特征检测的代码 !important

if __name__ == "__main__":
    img = cv2.imread('../data/images/lena.jpg')  # H,W,C
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # H,W,C->H,W


    #* Harris角点特征检测
    '''
    dst = cv2.cornerHarris(src, blockSize, ksize, k)
    '''
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)  # H,W->H,W
    print(gray.shape,'->',dst.shape)

    img[dst>0.01*dst.max()]=[255,0,0]  # # 自适应阈值;H,W,C->H,W,C
    cv_show(img)


    #* SIFT关键点特征检测
    '''
    input.shape:H,W
    output.shape:N,D
    where N is the number of keypoints and D is the feature dimension
    '''
    sift = cv2.SIFT_create()  # <class 'cv2.SIFT'>
    kp = sift.detect(gray, mask=None)  # N个关键点:<class 'cv2.KeyPoint'>
    printTupleShape(kp)
    # 输入图像,关键点列表,输出图像
    img_kp = cv2.drawKeypoints(gray, kp, img)  # H,W / N,KeyPoint / H,W,C->H,W,C
    cv_show(img_kp)
    
    # 计算关键点的描述子
    kp, des = sift.compute(gray, kp)  # H,W / N,KeyPoint -> N / N,D 
    print (np.array(kp).shape,des.shape)  # N / N,D


    ## 练习:
    # (1)角点检测
    img = cv2.imread('../data/images/lena.jpg')  # H,W,C
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # H,W,C->H,W
    dst = cornerHarris(gray, 3, 3, 0.04)  # H,W->H,W
    img[dst>0.01*dst.max()]=[255,0,0]  # # 自适应阈值;H,W,C->H,W,C
    cv_show(img)

    # (2)关键点检测
    # To be done:Keypoint Detection