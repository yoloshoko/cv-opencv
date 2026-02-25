import cv2
import numpy as np 

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