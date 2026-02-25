import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from utils.data_utils import *

# 求得的结果是边缘(即变化最大的地方)
# 这三个算子的终极目的都是为了检测图像中的边缘。Sobel和Scharr算子都是图片的一阶导，Laplacian是图片的二阶导
# 但是Scharr算子对结果更敏感，而Laplacian算子对变化更敏感

if __name__ == "__main__":
    img = cv2.imread("../data/images/pie.png")  # H,W,C

    #* Sobel算子(G)
    sobelxy = sobel_gradient(img)
    cv_show(sobelxy,'sobelxy')

    
    #* Scharr算子(G)
    scharrxy = scharr_gradient(img)
    cv_show(scharrxy,'scharrxy')


    #* Laplacian算子(G)
    laplacian = laplacian_gradient(img)
    cv_show(laplacian,'laplacian')

    