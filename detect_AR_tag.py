#!/usr/bin/python3
from turtle import color
import cv2
import numpy as np
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt
import pandas as pd

from utils import *


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
    f_ishift = np.fft.ifftshift(fshift)

    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    lower_thresh = np.min(img_back)
    upper_thresh = np.max(img_back)
    img_back = (img_back - lower_thresh)/upper_thresh

    return img_back


def circular_mask(img):
    img_copy = img.copy()
    img = np.uint8(img *255)
    img[:,0:10] = 0
    img[:,-10:] = 0
    img[0:10,:] = 0
    img[-10:,:] = 0
    mask = np.zeros_like(img,dtype='uint8')
         
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

    return masked_img


def get_corners(img,ct,num_corners,quality,distance):
    img_copy = img.copy()
    shi_tomasi_corners = np.int0(cv2.goodFeaturesToTrack(np.float32(img),num_corners,quality,distance))
   
    for corners in shi_tomasi_corners:
        x,y = corners.ravel()
        cv2.circle(img_copy,(x,y),10,255,-1)

    return img_copy,shi_tomasi_corners

def find_extreme_values(corners, masked_img):
    img = masked_img.copy()

    corners = corners.reshape(corners.shape[0],2)
    pt1 = corners[np.argmin(corners[:,0])]
    pt2 = corners[np.argmin(corners[:,1])]
    pt3 = corners[np.argmax(corners[:,0])]
    pt4 = corners[np.argmax(corners[:,1])]

    cv2.circle(img, pt1,5,(255,255,255),-1)
    cv2.circle(img, pt2,5,(0,255,0),-1)
    cv2.circle(img, pt3,5,(255,0,255),-1)
    cv2.circle(img, pt4,5,(255,255,0),-1)

    return img,pt1,pt2,pt3,pt4


def remove_outer_edges(masked_img,pt1,pt2,pt3,pt4):
    pts = np.array([pt1,pt2,pt3,pt4])
    outer_padding = 100
    pt_o1 = np.array([pt1[0] - outer_padding,pt1[1] + outer_padding])
    pt_o2 = np.array([pt2[0] - outer_padding,pt2[1] - outer_padding])
    pt_o3 = np.array([pt3[0] + outer_padding,pt3[1] - outer_padding])
    pt_o4 = np.array([pt4[0] + outer_padding,pt4[1] + outer_padding])
    pts_o = np.array([pt_o1,pt_o2,pt_o3,pt_o4])

    img = masked_img.copy()
    img_1 = masked_img.copy()

    cv2.polylines(img,[pts],True,(0,0,0),80)
    cv2.polylines(img_1,[pts],True,(0,0,0),80)
    cv2.polylines(img_1,[pts_o],True,(0,0,0),outer_padding+50)

    return img,img_1

def decode_tag(img):
    color = np.dstack((img,img,img))
    color_copy = color.copy()
    cv2.line(color,(200,0),(200,400),(0,0,255),3,cv2.LINE_AA)
    cv2.line(color,(100,0),(100,400),(0,0,255),3,cv2.LINE_AA)
    cv2.line(color,(300,0),(300,400),(0,0,255),3,cv2.LINE_AA)

    cv2.line(color,(0,200),(400,200),(0,0,255),3,cv2.LINE_AA)
    cv2.line(color,(0,100),(400,100),(0,0,255),3,cv2.LINE_AA)
    cv2.line(color,(0,300),(400,300),(0,0,255),3,cv2.LINE_AA)

    inner_box = color_copy[100:300,100:300]
    cv2.line(inner_box,(0,100),(200,100),(0,0,255),3,cv2.LINE_AA)
    cv2.line(inner_box,(0,50),(200,50),(0,0,255),3,cv2.LINE_AA)
    cv2.line(inner_box,(0,150),(200,150),(0,0,255),3,cv2.LINE_AA)

    cv2.line(inner_box,(100,0),(100,200),(0,0,255),3,cv2.LINE_AA)
    cv2.line(inner_box,(50,0),(50,200),(0,0,255),3,cv2.LINE_AA)
    cv2.line(inner_box,(150,0),(150,200),(0,0,255),3,cv2.LINE_AA)
    lt_corner = inner_box[:50,:50]
    lb_corner = inner_box[150:,:50]
    rt_corner = inner_box[:50,150:]
    rb_corner = inner_box[150:,150:]
    lt_whites = np.argwhere(lt_corner==(255,255,255)).shape[0]
    lb_whites = np.argwhere(lb_corner==(255,255,255)).shape[0]
    rt_whites = np.argwhere(rt_corner==(255,255,255)).shape[0]
    rb_whites = np.argwhere(rb_corner==(255,255,255)).shape[0]

    vals = np.array([lt_whites,rt_whites,rb_whites,lb_whites])
    offset = np.argmax(vals) # clockwise rotation offset starting from left top

    tag_id_area = inner_box[50:150,50:150]
    tag_id_area = cv2.cvtColor(tag_id_area,cv2.COLOR_BGR2GRAY)
    tag_img = tag_id_area.copy()
   
    tag_lt = tag_id_area[:50,:50]
    tag_rt = tag_id_area[:50,50:]
    tag_lb = tag_id_area[50:,:50]
    tag_rb = tag_id_area[50:,50:]
    tag_lt_whites = np.argwhere(tag_lt==255).shape[0]
    tag_lb_whites = np.argwhere(tag_lb==255).shape[0]
    tag_rt_whites = np.argwhere(tag_rt==255).shape[0]
    tag_rb_whites = np.argwhere(tag_rb==255).shape[0]
    tag_corners = [tag_lt_whites,tag_rt_whites,tag_rb_whites,tag_lb_whites]

    encoded_values =[]
    for i in range(4):
        if(tag_corners[i]>1800):
            encoded_values.append(1)
        else:
            encoded_values.append(0)

    tag_id = 0
    for i in range(4):
        tag_id += 2**i*encoded_values[(i+offset)%4]
    
    return color, inner_box,tag_id,offset,tag_img

def main(video_path):
    retval = True
    cap = cv2.VideoCapture(video_path)

    
    while(True):
        retval, frame = cap.read()   

        if(not retval):
            print("Video Completed.")
            break  

        img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     
        ret,thresh = cv2.threshold(img_gray,150,255,cv2.THRESH_BINARY)
 
        img_fft = fast_fourier_transform(thresh)

        masked_img = circular_mask(img_fft)
        
        masked_gray = cv2.cvtColor(masked_img,cv2.COLOR_BGR2GRAY)
        all_corner_img,corners = get_corners(masked_gray,-1,25,0.01,70)
    
        outer_corners_img,pt1,pt2,pt3,pt4 = find_extreme_values(corners, masked_img)

        inner_img,inner_img_1 = remove_outer_edges(masked_img,pt1,pt2,pt3,pt4)

        inner_gray = cv2.cvtColor(inner_img_1,cv2.COLOR_BGR2GRAY)
        all_inner_corners,corners = get_corners(inner_gray,-1,50,0.00001,1)
        inner_corners_img,ipt1,ipt2,ipt3,ipt4 = find_extreme_values(corners, inner_img_1)
        corners = np.array([ipt2,ipt3,ipt4,ipt1])
        
        H = final_homography_matrix(corners)

        warped = warpPerspective(thresh, H)

        decoded,inner_tag,tag_id,offset,tag_img = decode_tag(warped)
        offset_map = ['lt','rt','rb','lb']
        # cv2.putText(inner_tag,offset_map[offset]+",id:"+str(tag_id),(60,60),cv2.FONT_HERSHEY_SIMPLEX, 1, 255,5)
        cv2.imshow('tag',tag_img)
        print("tag_id:",tag_id)
        cv2.putText(frame,"Tag ID: " + str(tag_id),(80,80),cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,255),5)
        cv2.imshow('final_frame',frame)
        cv2.waitKey(1)
       
if __name__=="__main__":
    video_path = './media/1tagvideo.mp4'
    main(video_path)

    