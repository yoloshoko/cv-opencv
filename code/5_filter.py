import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from utils.data_utils import *



if __name__ == "__main__":
    img = cv2.imread("../data/images/lena_noise.png")
    img = cv2.resize(img,(300,300))

    blur = cv2.blur(img, (3, 3)) 
    # blur <=> box1
    box1 = cv2.boxFilter(img,-1,(3,3), normalize=True)  
    
    box2 = cv2.boxFilter(img,-1,(3,3), normalize=False)  
    aussian = cv2.GaussianBlur(img, (5, 5), 1)  
    median = cv2.medianBlur(img, 5)
    res = np.hstack((img,blur,box1,box2,aussian,median))
    cv_show(res)