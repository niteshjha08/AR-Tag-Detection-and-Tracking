#!/usr/bin/python3
from turtle import color
import cv2
import numpy as np
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt

def fast_fourier_transform(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    mag_specturm = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    h, w = img.shape

    center_y, center_x = int(h / 2), int(w / 2)

    mask = np.ones((h, w, 2), np.uint8)

    radius = 300

    y, x = np.ogrid[:h, :w]

    mask_area = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

    mask[mask_area] = 0

    fshift = dft_shift * mask

    fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
    # fshift_mask_mag = 2000 * np.log(np.abs(fshift[0,:],fshift[1,:]))


    f_ishift = np.fft.ifftshift(fshift)

    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    lower_thresh = np.min(img_back)
    upper_thresh = np.max(img_back)
    img_back = (img_back - lower_thresh)/upper_thresh
    return img_back

def find_extreme_values(corners, masked_img):
    img = masked_img.copy()

    corners = corners.reshape(corners.shape[0],2)
    pt1 = corners[np.argmin(corners[:,0])]
    pt2 = corners[np.argmin(corners[:,1])]
    pt3 = corners[np.argmax(corners[:,0])]
    pt4 = corners[np.argmax(corners[:,1])]

    cv2.circle(img, pt1,5,(0,0,255),-1)
    cv2.circle(img, pt2,5,(0,255,0),-1)
    cv2.circle(img, pt3,5,(255,0,255),-1)
    cv2.circle(img, pt4,5,(255,255,0),-1)

    cv2.namedWindow('end_corners',cv2.WINDOW_KEEPRATIO) 
    cv2.imshow('end_corners',img)
    return pt1,pt2,pt3,pt4

def circular_mask(img):
    img_copy = img.copy()
    img = np.uint8(img *255)
    img[:,0:10] = 0
    img[:,-10:] = 0
    img[0:10,:] = 0
    img[-10:,:] = 0
    mask = np.zeros_like(img,dtype='uint8')

    cv2.namedWindow('masked_img',cv2.WINDOW_KEEPRATIO) 
    cv2.namedWindow('mask_viz',cv2.WINDOW_KEEPRATIO)              

    # ret, img = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
    whites = np.argwhere(img>100)
    whites_x, whites_y = whites[:,0],whites[:,1]
    center = [np.int(np.mean(whites_x)),np.int(np.mean(whites_y))]
    radius = 450
    color_img = np.dstack((img,img,img))

    x= np.arange((mask.shape[1]))
    y= np.arange((mask.shape[0]))
    xx, yy = np.meshgrid(x,y)
    z = (xx-center[1])**2 + (yy-center[0])**2 - radius**2
    mask[z<0] = 255
    mask_3d = np.dstack((mask,mask,mask))
    masked_img = cv2.bitwise_and(color_img,mask_3d)
    masked_copy=masked_img.copy()
    cv2.circle(masked_copy,(center[1],center[0]),500,255,4)
    cv2.imshow('masked_img',masked_img)
    cv2.imshow('mask_viz',masked_copy)


    return masked_img


def get_corners(img,ct):
    img_copy = img.copy()
    shi_tomasi_corners = np.int0(cv2.goodFeaturesToTrack(np.float32(img),25,0.01,70))
    # return corners
    for corners in shi_tomasi_corners:
        x,y = corners.ravel()
        cv2.circle(img_copy,(x,y),10,255,-1)
        # print(shi_tomasi_corners.shape) 
    cv2.imshow('corners',img_copy) 
    print(img_copy.shape)  
    cv2.putText(img_copy, str(ct), (50,110), cv2.FONT_HERSHEY_SIMPLEX, 5, 255,10)
    cv2.imshow("corners",img_copy)
    return shi_tomasi_corners

def main(video_path):
    retval = True
    cap = cv2.VideoCapture(video_path)
    ct = 0
    cv2.namedWindow('img_gray',cv2.WINDOW_KEEPRATIO)

    cv2.namedWindow('fft',cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('thresh',cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('corners',cv2.WINDOW_KEEPRATIO)

    while(True):
        retval, frame = cap.read()            
        img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('img_gray',img_gray)
        ret,thresh = cv2.threshold(img_gray,150,255,cv2.THRESH_BINARY)
        cv2.imshow('thresh',thresh)
        img_fft = fast_fourier_transform(thresh)
        cv2.imshow('fft',img_fft)

        masked_img = circular_mask(img_fft)
        
        masked_gray = cv2.cvtColor(masked_img,cv2.COLOR_BGR2GRAY)
        corners = get_corners(masked_gray,ct)
        pts1,pt2,pt3,pt4 = find_extreme_values(corners, masked_img)
        cv2.waitKey(0)
        # break            

        ct+=1
        

if __name__=="__main__":
    video_path = './media/1tagvideo.mp4'
    main(video_path)