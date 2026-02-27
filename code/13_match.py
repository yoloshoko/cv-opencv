import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from utils.data_utils import *

### 1对1/1对多 特征匹配
# (1)1对1匹配：通过暴力匹配去计算SIFT检测的关键点之间的距离来衡量
# (2)1对多匹配：借鉴于KNN的分类思想进行求解,得到两个多组匹配,最后通过最邻点/次邻点的比值进行过滤


if __name__ == "__main__":
    img1 = cv2.imread('../data/images/box.png', 0)  # H,W,C->H,W
    img2 = cv2.imread('../data/images/box_in_scene.png', 0)  # H,W,C->H,W
    # cv_show(img1); cv_show(img2)
    
    
    #* SIFT关键点特征检测
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)  # H,W->(N,keypoint) / (N,D)
    kp2, des2 = sift.detectAndCompute(img2, None)  # H,W->(N,keypoint) / (N,D)


    #* 1对1匹配(暴力匹配)
    bf = cv2.BFMatcher(crossCheck=True)  # 相互匹配
    matches = bf.match(des1, des2)  # N,D and N,D -> M,<class 'cv2.DMatch'>
    # printTupleShape(matches)
    matches = sorted(matches, key=lambda x: x.distance)  
    # print(matches[0].distance)
    '''
    cv2.drawMatches(img1, kp1, img2, kp2, matches, outImg, flags)
    '''
    # height = max(img1.shape[0], img2.shape[0])
    # width = img1.shape[1] + img2.shape[1]
    # out_img = np.zeros((height, width), dtype=np.uint8)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None,flags=2)  # 在输出图像上绘制
    # print(img1.shape,img2.shape,'->',img3.shape)
    # cv_show(img3)


    #* k对最佳匹配(knn匹配)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    printTupleShape(matches[0])  # N,2,<class 'cv2.DMatch'>
    good = [] 
    for m, n in matches:  # m是最邻点,n是次邻点(m<n),目的是筛选出独特的匹配
        if m.distance < 0.5 * n.distance:
            good.append([m])

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
    # cv_show(img3)


