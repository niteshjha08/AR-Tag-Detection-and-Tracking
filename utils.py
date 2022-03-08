#!/usr/bin/python3
import numpy as np
import math
import cv2


# def warpPoint(H,pt):

#     pt = np.array([pt[0],pt[1],1])
#     # print(pt)
#     ind = np.indices((4,5))
#     x = np.arange(4)
#     y = np.arange(5)
#     xx,yy = np.meshgrid(x,y)


def final_homography_matrix(corners):
    xw1,yw1 = corners[0]
    xw2,yw2 = corners[1]
    xw3,yw3 = corners[2]
    xw4,yw4 = corners[3]

    xc1, yc1 = 0,0
    xc2 , yc2 = 400,0
    xc3 , yc3 = 400,400
    xc4 , yc4 = 0,400

    A = np.array([[xw1, yw1, 1, 0, 0, 0, -xc1*xw1, -xc1*yw1, -xc1],
         [0, 0, 0, xw1, yw1, 1, -yc1*xw1, -yc1*yw1, -yc1],
         [xw2, yw2, 1, 0, 0, 0, -xc2*xw2, -xc2*yw2, -xc2],
         [0, 0, 0, xw2, yw2, 1, -yc2*xw2, -yc2*yw2, -yc2],
         [xw3, yw3, 1, 0, 0, 0, -xc3*xw3, -xc3*yw3, -xc3],
         [0, 0, 0, xw3, yw3, 1, -yc3*xw3, -yc3*yw3, -yc3],
         [xw4, yw4, 1, 0, 0, 0, -xc4*xw4, -xc4*yw4, -xc4],
         [0, 0, 0, xw4, yw4, 1, -yc4*xw4, -yc4*yw4, -yc4]])

    W1, U = np.linalg.eig(np.dot(A,A.T))

    W2,V=np.linalg.eig(np.dot(A.T,A))

    eigenW1_order = np.flip(np.argsort(W1))
    eigenW2_order = np.flip(np.argsort(W2))
    W1 = W1[eigenW1_order]
    W2 = W2[eigenW2_order]
    U = U[:, eigenW1_order]
    V = V[:, eigenW2_order]

    m,n=A.shape
    S = np.zeros((A.shape))
    for i in range(np.min(A.shape)):
        S[i,i] = np.sqrt(np.abs(W1[i]))

    H=V[:, V.shape[1] - 1].reshape((3,3))

    c = H[2,2]
    H/=c

    return H

def final_homography_matrix_reverse(corners):
    xc1,yc1 = corners[0]
    xc2,yc2 = corners[1]
    xc3,yc3 = corners[2]
    xc4,yc4 = corners[3]

    xw1, yw1 = 0,0
    xw2 , yw2 = 400,0
    xw3 , yw3 = 400,400
    xw4 , yw4 = 0,400

    A = np.array([[xw1, yw1, 1, 0, 0, 0, -xc1*xw1, -xc1*yw1, -xc1],
         [0, 0, 0, xw1, yw1, 1, -yc1*xw1, -yc1*yw1, -yc1],
         [xw2, yw2, 1, 0, 0, 0, -xc2*xw2, -xc2*yw2, -xc2],
         [0, 0, 0, xw2, yw2, 1, -yc2*xw2, -yc2*yw2, -yc2],
         [xw3, yw3, 1, 0, 0, 0, -xc3*xw3, -xc3*yw3, -xc3],
         [0, 0, 0, xw3, yw3, 1, -yc3*xw3, -yc3*yw3, -yc3],
         [xw4, yw4, 1, 0, 0, 0, -xc4*xw4, -xc4*yw4, -xc4],
         [0, 0, 0, xw4, yw4, 1, -yc4*xw4, -yc4*yw4, -yc4]])

    W1, U = np.linalg.eig(np.dot(A,A.T))

    W2,V=np.linalg.eig(np.dot(A.T,A))

    eigenW1_order = np.flip(np.argsort(W1))
    W1 = W1[eigenW1_order]
    U = U[:, eigenW1_order]

    eigenW2_order = np.flip(np.argsort(W2))
    W2 = W2[eigenW2_order]
    V = V[:, eigenW2_order]

    m,n=A.shape
    S = np.zeros((A.shape))
    for i in range(np.min(A.shape)):
        S[i,i] = np.sqrt(np.abs(W1[i]))

    H=V[:, V.shape[1] - 1].reshape((3,3))

    c = H[2,2]
    H/=c

    return H

def warpPerspective(img, H):

    H_inv=np.linalg.inv(H)
    warped=np.zeros((400,400),np.uint8)
    
    for a in range(warped.shape[0]):
        for b in range(warped.shape[1]):
            hom_coord = [a,b,1]
            hom_coord = np.reshape(hom_coord,(3,1))
            x, y, z = np.matmul(H_inv,hom_coord)
            warped[a][b] = img[int(y/z)][int(x/z)]
    return warped

def warp_img_on_tag(template_img,src,H):

    H_inv = np.linalg.inv(H)

    for a in range(template_img.shape[1]):
        for b in range(template_img.shape[0]):
            hom_coord = [a,b,1]
            hom_coord = np.reshape(hom_coord,(3,1))
            x, y, z = np.matmul(H_inv,hom_coord)
            x,y = int(x/z),int(y/z)
           
            if(y>0 and y<1080 and x>0 and x<1920):
                src[y,x] = template_img[a][b]
    return src
    
def getProjectionMatrix(h, K):  
    h1 = h[:,0]          
    h2 = h[:,1]
    h3 = h[:,2]
   
    lamda = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(K),h1)) + np.linalg.norm(np.matmul(np.linalg.inv(K),h2)))
    b_t = lamda * np.matmul(np.linalg.inv(K),h)

    #check if determinant > 0 or otherwise, make it positive
    det = np.linalg.det(b_t)

    if det > 0:
        b = b_t
    else:                   
        b = -1 * b_t  
    row1 = b[:, 0]
    row2 = b[:, 1]
    t = b[:, 2]                     
    row3 = np.cross(row1, row2)
    
    Rt = np.column_stack((row1, row2, row3, t))
    P = np.matmul(K,Rt)  
    return(P,Rt,t)
    

def draw_cube_frame(frame,P):

    x1,y1,z1 = np.matmul(P,[0,0,0,1])
    x2,y2,z2 = np.matmul(P,[400,0,0,1])
    x3,y3,z3 = np.matmul(P,[400,400,0,1])
    x4,y4,z4 = np.matmul(P,[0,400,0,1])
    x5,y5,z5 = np.matmul(P,[0,0,-400,1])
    x6,y6,z6 = np.matmul(P,[400,0,-400,1])
    x7,y7,z7 = np.matmul(P,[400,400,-400,1])
    x8,y8,z8 = np.matmul(P,[0,400,-400,1])
    

    cv2.line(frame,(int(x1/z1),int(y1/z1)),(int(x5/z5),int(y5/z5)), (255,0,0), 2)
    cv2.line(frame,(int(x2/z2),int(y2/z2)),(int(x6/z6),int(y6/z6)), (255,0,0), 2)
    cv2.line(frame,(int(x3/z3),int(y3/z3)),(int(x7/z7),int(y7/z7)), (255,0,0), 2)
    cv2.line(frame,(int(x4/z4),int(y4/z4)),(int(x8/z8),int(y8/z8)), (255,0,0), 2)

    cv2.line(frame,(int(x1/z1),int(y1/z1)),(int(x2/z2),int(y2/z2)), (0,255,0), 2)
    cv2.line(frame,(int(x1/z1),int(y1/z1)),(int(x3/z3),int(y3/z3)), (0,255,0), 2)
    cv2.line(frame,(int(x2/z2),int(y2/z2)),(int(x4/z4),int(y4/z4)), (0,255,0), 2)
    cv2.line(frame,(int(x3/z3),int(y3/z3)),(int(x4/z4),int(y4/z4)), (0,255,0), 2)

    cv2.line(frame,(int(x5/z5),int(y5/z5)),(int(x6/z6),int(y6/z6)), (0,0,255), 2)
    cv2.line(frame,(int(x5/z5),int(y5/z5)),(int(x7/z7),int(y7/z7)), (0,0,255), 2)
    cv2.line(frame,(int(x6/z6),int(y6/z6)),(int(x8/z8),int(y8/z8)), (0,0,255), 2)
    cv2.line(frame,(int(x7/z7),int(y7/z7)),(int(x8/z8),int(y8/z8)), (0,0,255), 2)
    cv2.imshow("CUBE_VIDEO", frame)


    
