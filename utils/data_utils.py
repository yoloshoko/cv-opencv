import cv2
import numpy as np 
from collections import deque


def cv_show(img,name="img"):
    cv2.imshow(name,img)
    cv2.waitKey()  # 参数0等待任何键，27等价ESC键[默认是0] /也可以是等待的秒数
    cv2.destroyAllWindows()


def sim_img(bg=0):
    img = np.full((500, 500, 3), bg, dtype=np.uint8)  # 黑色图片
    return img


def video_show(vc):
    if vc.isOpened(): 
        open, frame = vc.read()
    else:
        open = False
    while open:
        ret, frame = vc.read()  # <class 'bool'>,(H,W,C)
        if frame is None:
            break
        if ret == True:
            gray = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)
            cv2.imshow('result', gray)
            if cv2.waitKey(100) & 0xFF == 27:  # 不是ESC继续执行
                break
    vc.release()
    cv2.destroyAllWindows()


def show_1channel(img,channel_idx=0):
    cur_img = img.copy()
    if channel_idx == 0:
        cur_img[:,:,1]=0
        cur_img[:,:,2]=0
    elif channel_idx == 1:
        cur_img[:,:,0]=0
        cur_img[:,:,2]=0
    elif channel_idx == 2:
        cur_img[:,:,0]=0
        cur_img[:,:,1]=0
    return cur_img


def sobel_gradient(img,direct=False):
    if direct == False:
        ## |Gx|
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)  # H,W,C->H,W,C
        print("sobelx:",type(sobelx))  # <class 'numpy.ndarray'>
        sobelx = cv2.convertScaleAbs(sobelx)  # H,W,C->H,W,C
        print("sobelx:",type(sobelx))  # <class 'numpy.ndarray'>
        # print(cv2.CV_64F,sobelx.shape)  # 常量:6,表示浮点数
        # cv_show(sobelx)

        ## |Gy|
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)  # H,W,C->H,W,C
        sobely = cv2.convertScaleAbs(sobely)  # H,W,C->H,W,C  
        # print(cv2.CV_64F,sobely.shape)  # 常量:6,表示浮点数
        # cv_show(sobely)

        ## |Gx|+|Gy|
        sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)  # H,W,C->H,W,C
    else:
        sobelxy=cv2.Sobel(img,cv2.CV_64F,1,1,ksize=3)  # H,W,C->H,W,C
        sobelxy = cv2.convertScaleAbs(sobelxy)  # H,W,C->H,W,C 
        # cv_show(sobelxy,'sobelxy')
    return sobelxy


def scharr_gradient(img,direct=False):
    if direct == False:
        # 核固定为 3x3
        scharrx = cv2.Scharr(img,cv2.CV_64F,1,0)  # H,W,C->H,W,C
        scharry = cv2.Scharr(img,cv2.CV_64F,0,1)  # H,W,C->H,W,C
        scharrx = cv2.convertScaleAbs(scharrx)  # H,W,C->H,W,C   
        scharry = cv2.convertScaleAbs(scharry)  # H,W,C->H,W,C  
        scharrxy =  cv2.addWeighted(scharrx,0.5,scharry,0.5,0)  # H,W,C->H,W,C 
    else:
        scharrxy = cv2.Scharr(img,cv2.CV_64F,1,1)  # H,W,C->H,W,C
        scharrxy = cv2.convertScaleAbs(scharrxy)  # H,W,C->H,W,C

    return scharrxy


def laplacian_gradient(img):
    # 二阶导，不需要进行x和y方向的梯度的求解
    laplacian = cv2.Laplacian(img,cv2.CV_64F)  # H,W,C->H,W,C
    laplacian = cv2.convertScaleAbs(laplacian)  # H,W,C->H,W,C

    return laplacian 


def canny(img,low_thr,high_thr,kernel_size=3,sigma=1.0):
    '''
    input.shape:[H,W,C] or [H,W]
    output.shape:[H,W]
    '''
    # print(img.shape)
    if img.ndim == 3:  # not img.size
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # H,W,C->H,W
    ## 1.高斯滤波去噪
    img_gau = cv2.GaussianBlur(img,(kernel_size,kernel_size),sigma)  # H,W->H,W

    ## 2.计算梯度大小与方向

    ### |Gx|
    sobelx = cv2.Sobel(img_gau,cv2.CV_64F,1,0,ksize=kernel_size)  # H,W->H,W
    # sobelx_ = cv2.convertScaleAbs(sobelx)
    ### |Gy|
    sobely = cv2.Sobel(img_gau,cv2.CV_64F,0,1,ksize=kernel_size)  # H,W->H,W
    # sobely_ = cv2.convertScaleAbs(sobely)
    ### G=|Gx|+|Gy|
    # sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
    sobelxy = np.sqrt(sobelx**2 + sobely**2)  # H,W->H,W
    # print(sobelxy.shape)

    ### \theta = arctan(|Gy|/|Gx|)
    theta = np.arctan2(sobely,sobelx)  # H,W->H,W
    # print(type(theta),theta.shape)
    # print(theta.max(),theta.min()) range(-pi,pi)


    ## 3.非极大值抑制
    def no_max_suppress(sobelxy,theta):
        h,w = sobelxy.shape
        suppressed = np.zeros_like(sobelxy)

        angle = ( np.rad2deg(theta) + 180 ) % 180 # range(-pi,pi) -> range(-180,180)
        
        for i in range(1,h-1):
            for j in range(1,w-1):
                if( 0<= angle[i,j] < 22.5) or (157.5 <= angle[i,j]<=180):
                    neighbor1 = sobelxy[i-1,j]
                    neighbor2 = sobelxy[i+1,j]
                elif 22.5<=angle[i,j]<67.5:
                    neighbor1 = sobelxy[i-1,j-1]
                    neighbor2 = sobelxy[i+1,j+1]
                elif 67.5<=angle[i,j] <112.5:
                    neighbor1 = sobelxy[i,j-1]
                    neighbor2 = sobelxy[i,j+1]
                elif 112.5<=angle[i,j]<157.5:
                    neighbor1 = sobelxy[i-1,j+1]
                    neighbor2 = sobelxy[i+1,j-1]
                if sobelxy[i,j] >= neighbor1 and sobelxy[i,j] >= neighbor2:
                    suppressed[i,j] = sobelxy[i,j]

        return suppressed
    suppressed = no_max_suppress(sobelxy,theta)  # H,W->H,W

    ## 4.双阈值检测
    def double_threshold(img,low_thr,high_thr):
        h,w = img.shape
        
        # 强边缘、弱边缘、非边缘分类
        strong = 255
        weak = 50
        none = 0
        result = np.zeros_like(img,dtype=np.uint8)

        strong_i,strong_j=[],[]
        for i in range(h):
            for j in range(w):
                if img[i,j] >=high_thr:
                    result[i,j] = strong
                    strong_i.append(i)
                    strong_j.append(j)
                elif img[i,j]>=low_thr:
                    result[i,j] = weak

        direction = [(-1,-1),(-1,0),(-1,1),
                     (0,-1),(0,1),
                     (1,-1),(1,0),(1,1)]
        queue = deque(zip(strong_i,strong_j))

        while queue:
            x,y = queue.popleft()
            for dx,dy in direction:
                nx, ny = x+dx, y+dy
                if 0<=nx <h and 0<= ny <w:
                    if result[nx,ny] == weak:
                        result[nx,ny] = strong
                        queue.append((nx,ny))
        result[result == weak]=0
        return result
    edges = double_threshold(suppressed,low_thr,high_thr)
    print(edges.shape)
    return edges



def pyrDown(img):
    kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ], dtype=np.float32) / 256.0
    img = cv2.filter2D(img, -1, kernel)
    # img = cv2.GaussianBlur(img,(5,5),1)
    # 变小
    result = img[::2,::2]
    return result


def pyUp(img):
    kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ], dtype=np.float32) / 64.0
    # 变大
    h,w,c = img.shape
    upsampled = np.zeros((2*h,2*w,c), dtype=np.uint8)
    upsampled[::2,::2] = img

    result = np.zeros_like(upsampled)
    for idx_c in range(img.shape[2]):
        result[:,:,idx_c] = cv2.filter2D(upsampled[:,:,idx_c],-1,kernel)
    return result


def printTupleShape(datas):
    print(f"len(datas)={len(datas)}")
    for i, data in enumerate(datas):
        try:
            print(f"data {i}: {data.shape}")
        except:
            print(f"type(data):{type(data)}")
            break


def cornerHarris(img,block_size=3, ksize=3, k=0.04):
    if img.ndim == 3:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = np.float32(gray)
    h, w = gray.shape
    Ix = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=ksize)
    Iy = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=ksize)
    
    Ixx = Ix * Ix 
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    # print(Ixx.shape,Iyy.shape,Ixy.shape)

    Sxx = cv2.GaussianBlur(Ixx,(block_size,block_size),0)
    Syy = cv2.GaussianBlur(Iyy,(block_size,block_size),0)
    Sxy = cv2.GaussianBlur(Ixy,(block_size,block_size),0)

    # 初始化相应矩阵
    det_M = Sxx * Syy - Sxy ** 2
    trace_M = Sxx + Syy 
    R = det_M - k * (trace_M ** 2)
    return R

