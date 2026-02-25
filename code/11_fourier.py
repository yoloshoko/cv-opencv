import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from utils.data_utils import *


### 傅里叶变换的作用
# 高频：变化剧烈的灰度分量，例如边界
# 低频：变化缓慢的灰度分量，例如一片大海

### 滤波
# 低通滤波器：只保留低频，会使得图像模糊
# 高通滤波器：只保留高频，会使得图像细节增强


if __name__ == "__main__":
    img = cv2.imread('../data/images/lena.jpg',0)  # H,W,C -> H,W ;0:以灰度图模式读取图像
    img_float32 = np.float32(img)

    #* 傅里叶分解:将时域转换成频域
    # return -> 两个通道:第1通道是实部，第2通道是虚部
    dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)  # 只接受H,W形状的灰度图; H,W -> H,W,2
    dft_shift = np.fft.fftshift(dft)  # 将低频移到中心; H,W -> H,W,2 
    print(img.shape,'->',dft.shape)
    print(img.shape,'->',dft_shift.shape)

    # A = 20 * log (sqrt(x ** 2 + y ** 2)),x和y分别是实部和虚部
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))  # H,W,2 -> H,W
    print(magnitude_spectrum.shape)

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


    def filter(filter_type=0):  # 0表示保留低频,1表示保留高频
        #* 低通/高通滤波器
        rows, cols = img.shape
        crow, ccol = int(rows/2) , int(cols/2)     # 中心位置
        if filter_type == 0:
            # 低通滤波
            mask = np.zeros((rows, cols, 2), np.uint8)  # H,W,2
            mask[crow-30:crow+30, ccol-30:ccol+30] = 1  # H,W,2
        else:
            mask = np.ones((rows, cols, 2), np.uint8)  # H,W,2
            mask[crow-30:crow+30, ccol-30:ccol+30] = 0  # H,W,2
        # IDFT
        fshift = dft_shift*mask  # H,W,2 @ H,W,2 -> H,W,2
        f_ishift = np.fft.ifftshift(fshift)  # 将高频返回中心;H,W,2 -> H,W,2
        img_back = cv2.idft(f_ishift)  # H,W,2-> H,W,2
        # A = sqrt(x ** 2 + y ** 2)
        img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])  # H,W,2->H,W

        plt.subplot(121),plt.imshow(img, cmap = 'gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
        plt.title('Result'), plt.xticks([]), plt.yticks([])
        plt.show()   

    filter(0)
    filter(1)


