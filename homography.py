#!/usr/bin/python3
import numpy as np
import math

def homography(x,y,xp,yp):
    # A=np.array([[-x[0],-y[0],-1,0,0,0,x[0]*xp[0],y[0]*xp[0],xp[0]],
    #             [0,0,0,-x[0],-y[0],-1,x[0]*yp[0],y[0]*yp[0],yp[0]],
    #             [-x[1],-y[1],-1,0,0,0,x[1]*xp[1],y[1]*xp[1],xp[1]],
    #             [0,0,0,-x[1],y[1],-1,x[1]*yp[1],y[1]*yp[1],yp[1]],
    #             [-x[2],-y[2],-1,0,0,0,x[2]*xp[2],y[2]*xp[2],xp[2]],
    #             [0,0,0,-x[2],y[2],-1,x[2]*yp[2],y[2]*yp[2],yp[2]],
    #             [-x[3],-y[3],-1,0,0,0,x[3]*xp[3],y[3]*xp[3],xp[3]],
    #             [0,0,0,-x[3],y[3],-1,x[3]*yp[3],y[3]*yp[3],yp[3]]])

    A=np.array([[x[0],y[0],1,0,0,0,-x[0]*xp[0],-y[0]*xp[0],-xp[0]],
                [0,0,0,x[0],y[0],1,-x[0]*yp[0],-y[0]*yp[0],-yp[0]],
                [x[1],y[1],1,0,0,0,-x[1]*xp[1],-y[1]*xp[1],-xp[1]],
                [0,0,0,x[1],y[1],1,-x[1]*yp[1],-y[1]*yp[1],-yp[1]],
                [x[2],y[2],1,0,0,0,-x[2]*xp[2],-y[2]*xp[2],-xp[2]],
                [0,0,0,x[2],y[2],1,-x[2]*yp[2],-y[2]*yp[2],-yp[2]],
                [x[3],y[3],1,0,0,0,-x[3]*xp[3],-y[3]*xp[3],-xp[3]],
                [0,0,0,x[3],-y[3],1,-x[3]*yp[3],-y[3]*yp[3],-yp[3]]])
    
    return A

def sort_eig_pairs(W,U):
    couple=[(W[i],U[:,i]) for i in range(len(W))]
    couple.sort(reverse=True)
    W=[couple[i][0] for i in range(len(couple))]
    U=[couple[i][1] for i in range(len(couple))]

    return np.array(W),np.array(U)

def get_homography_matrix(x,y,xp,yp):

    # x = np.array([pt1[0],pt2[0],pt3[0],pt4[0]])
    # y = np.array([pt1[1],pt2[1],pt3[1],pt4[1]])

    # xp=np.array([0,200,200,0])
    # yp=np.array([0,0,200,200])

    A=homography(x,y,xp,yp) #mxn=8x9

    # W1, U = np.linalg.eig(np.dot(A,A.T))

    # W1,U=sort_eig_pairs(W1,U)

    # W2,V=np.linalg.eig(np.dot(A.T,A))
    # W2,V=sort_eig_pairs(W2,V)

    # m,n=A.shape
    # S = np.zeros((A.shape))
    # for i in range(np.min(A.shape)):
    #     S[i,i] = np.sqrt(np.abs(W1[i]))

    # A_pred = np.dot(np.dot(U, S), V.T)

    # H=V.T[:,-1].reshape((3,3))
    # print("V_mine:\n",V)
    # c = H[2,2]
    # H/=c
    # return H

    u,S,Vh = np.linalg.svd(A)
    l= Vh[-1,:]/Vh[-1,-1]
    h= np.reshape(l,(3,3))
    # print("compute:\n",H)
    # print("svd:\n",h)
    return h

def warpPoint(H,pt):
    # print(pt.shape)
    # print(pt)
    pt = np.array([pt[0],pt[1],1])
    # print(pt)
    ind = np.indices((4,5))
    x = np.arange(4)
    y = np.arange(5)
    xx,yy = np.meshgrid(x,y)
    # print(ind)
    # print(yy)
    # print(xx)

def new_homography(corners):
    # def homography(corners,frame):
    xw1,yw1 = corners[0]
    xw2,yw2 = corners[1]
    xw3,yw3 = corners[2]
    xw4,yw4 = corners[3]

    # xw1,xw2,xw3,xw4 = x
    # yw1,yw2,yw3,yw4 = y
    # xc1,xc2,xc3,xc4 = xp
    # yc1,yc2,yc3,yc4 = yp
    
    xc1, yc1 = 0,0
    xc2 , yc2 = 400,0
    xc3 , yc3 = 400,800
    xc4 , yc4 = 0,800
    corners2 = np.array([
        [xc1,yc1],
        [xc2,yc2],
        [xc3,yc3],
        [xc4,yc4]
    ])
    A = np.array([[xw1, yw1, 1, 0, 0, 0, -xc1*xw1, -xc1*yw1, -xc1],
         [0, 0, 0, xw1, yw1, 1, -yc1*xw1, -yc1*yw1, -yc1],
         [xw2, yw2, 1, 0, 0, 0, -xc2*xw2, -xc2*yw2, -xc2],
         [0, 0, 0, xw2, yw2, 1, -yc2*xw2, -yc2*yw2, -yc2],
         [xw3, yw3, 1, 0, 0, 0, -xc3*xw3, -xc3*yw3, -xc3],
         [0, 0, 0, xw3, yw3, 1, -yc3*xw3, -yc3*yw3, -yc3],
         [xw4, yw4, 1, 0, 0, 0, -xc4*xw4, -xc4*yw4, -xc4],
         [0, 0, 0, xw4, yw4, 1, -yc4*xw4, -yc4*yw4, -yc4]])
    
    m, n = A.shape

    AA_t = np.dot(A, A.transpose())
    A_tA = np.dot(A.transpose(), A) 

    eigen_values_1, U = np.linalg.eig(AA_t)
    eigen_values_2, V = np.linalg.eig(A_tA)
    index_1 = np.flip(np.argsort(eigen_values_1))
    eigen_values_1 = eigen_values_1[index_1]
    U = U[:, index_1]
    index_2 = np.flip(np.argsort(eigen_values_2))
    eigen_values_2 = eigen_values_2[index_2]
    V = V[:, index_2]

    E = np.zeros([m, n])

    var = np.minimum(m, n)

    for j in range(var):
        E[j,j] = np.abs(np.sqrt(eigen_values_1[j]))  

    H = V[:, V.shape[1] - 1]
    # print("V_sai:\n",V)
    H = H.reshape([3,3])
    H = H / H[2,2]
    # print(Homography)
    # H_through_cv, _  = cv.findHomography(corners,corners2)
    # print(H_through_cv)
    return H


def final_homography_matrix(corners):
    xw1,yw1 = corners[0]
    xw2,yw2 = corners[1]
    xw3,yw3 = corners[2]
    xw4,yw4 = corners[3]

    # xw1,xw2,xw3,xw4 = x
    # yw1,yw2,yw3,yw4 = y
    # xc1,xc2,xc3,xc4 = xp
    # yc1,yc2,yc3,yc4 = yp
    
    xc1, yc1 = 0,0
    xc2 , yc2 = 400,0
    xc3 , yc3 = 400,400
    xc4 , yc4 = 0,400
    # xw1,xw2,xw3,xw4 = x
    # yw1,yw2,yw3,yw4 = y
    # xc1,xc2,xc3,xc4 = xp
    # yc1,yc2,yc3,yc4 = yp
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
    # print(img.shape)
    for a in range(warped.shape[0]):
        for b in range(warped.shape[1]):
            f = [a,b,1]
            f = np.reshape(f,(3,1))
            x, y, z = np.matmul(H_inv,f)
            # print(x,y,z)
            warped[a][b] = img[int(y/z)][int(x/z)]
    return(warped)
