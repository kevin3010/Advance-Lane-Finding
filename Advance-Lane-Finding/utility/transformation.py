import numpy as np
import cv2
import matplotlib.pyplot as plt

def remove_distortion(img,mtx,dist,display=False):
    undistort = cv2.undistort(img, mtx, dist, None, mtx)

    if display==True:
        fig,ax = plt.subplots(1,2)
        ax.flatten()

        ax[0].imshow(img)
        ax[1].imshow(undistort)

        plt.show()

        return undistort
    else:
        return undistort


def sobel_binary(img,orient='x',kernel = 3,thresh=(0,255),display=False):

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    if orient=='x':
        sobel = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=kernel)
    elif orient=='y':
        sobel = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=kernel)

    sobel = np.absolute(sobel)

    scale_factor = np.max(sobel) / 255
    sobel = np.uint8(sobel/scale_factor)

    binary_output = np.zeros_like(sobel)
    binary_output[(sobel>=thresh[0]) & (sobel<=thresh[1])] = 1 



    if display:
        # plt.imshow(img)

        fig, ax = plt.subplots(3,1)
        ax = ax.flatten()

        ax[0].imshow(img)
        ax[0].set(title='original')

        ax[1].imshow(binary_output,cmap='gray')
        ax[1].set(title='binary_output')

        ax[2].hist(sobel.ravel(),255,[0,255])
        ax[2].set(title='histogram')

        plt.show()

        return binary_output
    else:
        return binary_output

def mag_binary(img,kernel = 3,thresh=(0,255),display=False):

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=kernel)
    
    mag = np.sqrt(sobelx**2 + sobely**2)

    scale_factor = np.max(mag) / 255
    mag = np.uint8(mag/scale_factor)

    binary_output = np.zeros_like(mag)
    binary_output[(mag>=thresh[0]) & (mag<=thresh[1])] = 1 


    if display:
        # plt.imshow(img)

        fig, ax = plt.subplots(3,1)
        ax = ax.flatten()

        ax[0].imshow(img)
        ax[0].set(title='original')

        ax[1].imshow(binary_output,cmap='gray')
        ax[1].set(title='binary_output')

        ax[2].hist(mag.ravel(),255,[0,255])
        ax[2].set(title='histogram')

        plt.show()

        return binary_output
    else:
        return binary_output


def dir_binary(img,kernel=3,thresh=(0,np.pi/2),display=False):

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=kernel)
    
    direction = np.arctan2(np.absolute(sobely) , np.absolute(sobelx))

    binary_output = np.zeros_like(direction)
    binary_output[(direction>=thresh[0]) & (direction<=thresh[1])] = 1 


    if display:
        # plt.imshow(img)

        fig, ax = plt.subplots(3,1)
        ax = ax.flatten()

        ax[0].imshow(img)
        ax[0].set(title='original')

        ax[1].imshow(binary_output,cmap='gray')
        ax[1].set(title='binary_output')

        ax[2].hist(direction.ravel(),255,[0,255])
        ax[2].set(title='histogram')

        plt.show()

        return binary_output
    else:
        return binary_output

def color_binary(img,display=False):

    luv = cv2.cvtColor(img,cv2.COLOR_RGB2LUV)
    
    l_channel = luv[:,:,0]
    v_channel = luv[:,:,2]

    binary_output_l = np.zeros_like(l_channel)
    binary_output_l[(l_channel>=205) & (l_channel<255)] = 1

    binary_output_v = np.zeros_like(v_channel)
    binary_output_v[(v_channel>=160) & (v_channel<240)] = 1

    binary_output = np.zeros_like(v_channel)
    binary_output[(binary_output_v==1) | (binary_output_l==1)] = 1

    
    # binary_output = np.zeros_like(direction)
    # binary_output[(direction>=thresh[0]) & (direction<=thresh[1])] = 1 


    if display:
        # plt.imshow(img)

        fig, ax = plt.subplots(1,2)
        ax = ax.flatten()

        ax[0].imshow(img)
        ax[1].imshow(binary_output,cmap='gray')

        plt.show()

        return binary_output

    else:
        return binary_output

def perspective_transform(img,display = False):

    shape = (img.shape[1],img.shape[0])

    h = shape[0]
    w = shape[1]

    bottom_left = [300,683]
    bottom_right = [1120,683]
    top_left = [585,460]
    top_right = [705,460]

    pts = np.array([bottom_left,bottom_right,top_right,top_left], np.int32)

    copy = img.copy()
    cv2.polylines(copy,[pts],True,(255,0,0), thickness=1)

    tbl = [300,683]
    tbr = [1000,683]
    ttl = [300,0]
    ttr = [1000,0]

    src = np.float32([bottom_left,top_left,top_right,bottom_right])
    dst = np.float32([tbl,ttl,ttr,tbr])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, shape, flags=cv2.INTER_LINEAR)

    if display:
        # plt.imshow(img)

        fig, ax = plt.subplots(1,2)
        ax = ax.flatten()

        ax[0].imshow(copy)
        ax[0].set(title='original')

        ax[1].imshow(warped)
        ax[1].set(title='warped')

        plt.show()

    
    return Minv,warped

def combined_binary(img,display=False):

    sobel = sobel_binary(img,orient='x',kernel=7,thresh=(10,75),display=False) 
    mag = mag_binary(img,kernel = 3,thresh=(20,50),display=False)
    direction = dir_binary(img,kernel=7,thresh=(0,np.pi/4),display=False)
    color = color_binary(img,display=False)

    binary_output = np.zeros_like(direction)
    binary_output[((direction==1) & (mag==1)) | (color==1)] = 1
    binary_output[(color==1)] = 1

    if display:
        fig, ax = plt.subplots(3,2)
        ax = ax.flatten()

        ax[0].imshow(img)
        ax[1].imshow(binary_output,cmap='gray')
        ax[2].imshow(sobel,cmap='gray')
        ax[3].imshow(mag,cmap='gray')
        ax[4].imshow(direction,cmap='gray')
        ax[5].imshow(color,cmap='gray')
        plt.show()

        return binary_output
    else:
        return binary_output