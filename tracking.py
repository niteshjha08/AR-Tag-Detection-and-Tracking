#!/usr/bin/python3
from turtle import color
import cv2
import numpy as np
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt
import pandas as pd

from homography import *
from detect_AR_tag import *

def main(video_path,img_path):
    
    retval = True
    cap = cv2.VideoCapture(video_path)
    ct = 0
    cv2.namedWindow('img_gray',cv2.WINDOW_KEEPRATIO)

    # cv2.namedWindow('fft',cv2.WINDOW_KEEPRATIO)
    # cv2.namedWindow('thresh',cv2.WINDOW_KEEPRATIO)
    # cv2.namedWindow('corners',cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('outer_corners',cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('inner_corners',cv2.WINDOW_KEEPRATIO)
    # cv2.namedWindow('all_inner_corners',cv2.WINDOW_KEEPRATIO)
    # cv2.namedWindow('viz_padding',cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('warped',cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('decoded',cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('inner_tag',cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('tag',cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('testudo',cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('template_img',cv2.WINDOW_KEEPRATIO)


    
    while(True):                  
        retval, frame = cap.read()            
        img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('img_gray',img_gray)
        ret,thresh = cv2.threshold(img_gray,150,255,cv2.THRESH_BINARY)
        # cv2.imshow('thresh',thresh)
        img_fft = fast_fourier_transform(thresh)
        # cv2.imshow('fft',img_fft)

        masked_img = circular_mask(img_fft)
        
        masked_gray = cv2.cvtColor(masked_img,cv2.COLOR_BGR2GRAY)
        all_corner_img,corners = get_corners(masked_gray,ct,25,0.01,70)
        outer_corners_img,pt1,pt2,pt3,pt4 = find_extreme_values(corners, masked_img)
        cv2.imshow('outer_corners',outer_corners_img)
    
        inner_img,inner_img_1 = remove_outer_edges(masked_img,pt1,pt2,pt3,pt4)
        inner_gray = cv2.cvtColor(inner_img_1,cv2.COLOR_BGR2GRAY)
        all_inner_corners,corners = get_corners(inner_gray,ct,50,0.00001,1)
        # cv2.imshow('all_inner_corners',all_inner_corners)
        # cv2.imshow('viz_padding',inner_img_1)
        inner_corners_img,ipt1,ipt2,ipt3,ipt4 = find_extreme_values(corners, inner_img_1)
        cv2.imshow('inner_corners',inner_corners_img)
        corners = np.array([ipt2,ipt3,ipt4,ipt1])
        H = final_homography_matrix(corners)
        # print("my H:\n",H)
        # h_orig = cv2.getPerspectiveTransform(np.array([ipt2,ipt3,ipt4,ipt1],dtype=np.float32),np.array([[0,0],[400,0],[400,800],[0,800]],dtype=np.float32))
        # print("actual H:\n",h_orig)
        warped = warpPerspective(thresh, H)
        cv2.imshow('warped',warped)
        decoded,inner_tag,tag_id,offset,tag_img = decode_tag(warped)
        offset_map = ['lt','rt','rb','lb']
        cv2.putText(inner_tag,offset_map[offset]+",id:"+str(tag_id),(60,60),cv2.FONT_HERSHEY_SIMPLEX, 1, 255,5)
        template_img = get_image(img_path,offset)
        cv2.imshow('template',template_img)
        print(offset)
        # cv2.waitKey(0)
        final = warp_img_on_tag(template_img,frame,H)
        
        cv2.imshow('decoded',decoded)
        cv2.imshow('inner_tag',inner_tag)
        cv2.imshow('tag',tag_img)
        cv2.imshow('testudo',final)

        ct+=1
        cv2.waitKey(1)
        # break       

        

def get_image(image_path,offset):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(400,400))
    offset = (offset  + 1)%4
    num_clockwise_rotation = (4 - offset) % 4
    for i in range(num_clockwise_rotation):
        img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    return img

if __name__=="__main__":
    video_path = './media/1tagvideo.mp4'
    image_path = './media/testudo.png'
    # get_image(image_path,3) 
    main(video_path,image_path)
    # H = get_homography_matrix
    