#!/usr/bin/python3
from email import header
from turtle import color
import cv2
import numpy as np
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt
import pandas as pd

from utils import *
from detect_AR_tag import *


def get_image(image_path,offset):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(400,400))
    offset = (offset  + 1)%4
    num_cclockwise_rotation = (4 - offset) % 4
    for i in range(num_cclockwise_rotation):
        img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def get_camera_matrix(intrinsic_params_path):
    Kmatrix = np.array(pd.read_csv(intrinsic_params_path,header=None))
    return Kmatrix

def superpose_image(video_path,img_path):
    
    retval = True
    cap = cv2.VideoCapture(video_path)

    offset_map = ['lt','rt','rb','lb']
    video = cv2.VideoCapture(video_path)

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        fps = video.get(cv2.CAP_PROP_FPS)

    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4))
    vid = cv2.VideoWriter('./output/Testudo.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    while(True):                 
        retval, frame = cap.read() 
        if not retval:
            print("Video completed.")  
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
        
        template_img = get_image(img_path,offset)

        final = warp_img_on_tag(template_img,frame,H)
        
        cv2.imshow('testudo',final)
        vid.write(final)
        cv2.waitKey(1)      
    vid.release()
    cap.release()
    cv2.destroyAllWindows()


###################################################################
#                  Projecting a cube on the tag
###################################################################
def get_cube_points(P):
    P = np.reshape(P,(3,4))
    pts_cube = [[0,0,0,1],
                [400,0,0,1],
                [400,400,0,1],
                [0,400,0,1],
                [0,0,-400,1],
                [400,0,-400,1],
                [400,400,-400,1],
                [0,400,-400,1]]

    AR_cube_pts = []

    for i in range(8):
        pt = np.array(pts_cube[i]).reshape(4,1)

        res = np.matmul(P,pt)

        AR_cube_pts.append(res)

    return AR_cube_pts

def get_AR_cube(frame,AR_cube_pts):

    AR_cube_pts = np.array(AR_cube_pts)
    AR_cube_pts = np.reshape(AR_cube_pts,((8,3)))

    x = AR_cube_pts[:,0]
    y = AR_cube_pts[:,1]
    z = AR_cube_pts[:,2]


    cv2.line(frame,(int(x[0]/z[0]),int(y[0]/z[0])),(int(x[4]/z[4]),int(y[4]/z[4])), (255,0,0), 5,cv2.LINE_AA)
    cv2.line(frame,(int(x[1]/z[1]),int(y[1]/z[1])),(int(x[5]/z[5]),int(y[5]/z[5])), (255,0,0), 5)
    cv2.line(frame,(int(x[2]/z[2]),int(y[2]/z[2])),(int(x[6]/z[6]),int(y[6]/z[6])), (255,0,0), 5)
    cv2.line(frame,(int(x[3]/z[3]),int(y[3]/z[3])),(int(x[7]/z[7]),int(y[7]/z[7])), (255,0,0), 5)

    cv2.line(frame,(int(x[0]/z[0]),int(y[0]/z[0])),(int(x[1]/z[1]),int(y[1]/z[1])), (0,255,0), 5)
    cv2.line(frame,(int(x[1]/z[1]),int(y[1]/z[1])),(int(x[2]/z[2]),int(y[2]/z[2])), (0,255,0), 5)
    cv2.line(frame,(int(x[2]/z[2]),int(y[2]/z[2])),(int(x[3]/z[3]),int(y[3]/z[3])), (0,255,0), 5)
    cv2.line(frame,(int(x[3]/z[3]),int(y[3]/z[3])),(int(x[0]/z[0]),int(y[0]/z[0])), (0,255,0), 5)


    cv2.line(frame,(int(x[4]/z[4]),int(y[4]/z[4])),(int(x[5]/z[5]),int(y[5]/z[5])), (0,0,255), 5)
    cv2.line(frame,(int(x[5]/z[5]),int(y[5]/z[5])),(int(x[6]/z[6]),int(y[6]/z[6])), (0,0,255), 5)
    cv2.line(frame,(int(x[6]/z[6]),int(y[6]/z[6])),(int(x[7]/z[7]),int(y[7]/z[7])), (0,0,255), 5)
    cv2.line(frame,(int(x[4]/z[4]),int(y[4]/z[4])),(int(x[7]/z[7]),int(y[7]/z[7])), (0,0,255), 5)

    return frame


def place_AR_cube(video_path,K):
    cap = cv2.VideoCapture(video_path)
    ct=0
    cv2.namedWindow('frame',cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('img_gray',cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('inner_corners',cv2.WINDOW_KEEPRATIO)

    video = cv2.VideoCapture(video_path)
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4))
    vid = cv2.VideoWriter('./output/Cube.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
    while(True):   
        print("frame:",ct)               
        retval, frame = cap.read() 
        if not retval:
            print("Video completed.")  
            break  
        frame_copy = frame.copy()
        img_gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        ret,thresh = cv2.threshold(img_gray,150,255,cv2.THRESH_BINARY)

        img_fft = fast_fourier_transform(thresh)

        masked_img = circular_mask(img_fft)
        
        masked_gray = cv2.cvtColor(masked_img,cv2.COLOR_BGR2GRAY)
        all_corner_img,corners = get_corners(masked_gray,ct,25,0.01,70)
        outer_corners_img,pt1,pt2,pt3,pt4 = find_extreme_values(corners, masked_img)

        inner_img,inner_img_1 = remove_outer_edges(masked_img,pt1,pt2,pt3,pt4)
        inner_gray = cv2.cvtColor(inner_img_1,cv2.COLOR_BGR2GRAY)
        all_inner_corners,corners = get_corners(inner_gray,ct,50,0.00001,1)

        inner_corners_img,ipt1,ipt2,ipt3,ipt4 = find_extreme_values(corners, inner_img_1)

        corners = np.array([ipt2,ipt3,ipt4,ipt1])


        H = final_homography_matrix_reverse(corners)

        P,Rt,t = getProjectionMatrix(H,K)

        cube_pts = get_cube_points(P)
        
        cube_img = get_AR_cube(frame_copy,cube_pts)

        cv2.imshow('cube_img',cube_img) 
        cv2.waitKey(1)


if __name__=="__main__":
    video_path = './media/1tagvideo.mp4'
    image_path = './media/testudo.png'
    intrinsic_params_path = './param/kmatrix.csv'
    Superimpose_Testudo = False
    Place_AR_Cube = True
    K = get_camera_matrix(intrinsic_params_path)
    if Superimpose_Testudo:
        superpose_image(video_path,image_path)

    if Place_AR_Cube:
        place_AR_cube(video_path,K)
    