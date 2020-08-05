import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from glob import glob



paths = glob('./calibration_images/calibration*.jpg')

objpoints = [] 
imgpoints = []

for path in paths:
    print(path)

    img = cv2.imread(path)
    img = cv2.undistort(img, mtx, dist, None, mtx)
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    objp = np.zeros(((6*9),3),np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,corner = cv2.findChessboardCorners(gray,(9,6),None)

    if ret==True:
        objpoints.append(objp)
        imgpoints.append(corner)
        draw_img = cv2.drawChessboardCorners(img, (9, 6), corner, ret)
        
        file_name = path.split('\\')[-1]
        cv2.imwrite('./calibration_images/saved_images/'+file_name,draw_img)
        
        
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (720,1280), None, None)

np.save('mtx.npy',mtx)
np.save('dist.npy',dist)

# for path in paths:

#     img = cv2.imread(path)
#     dst = cv2.undistort(img, mtx, dist, None, mtx)

#     file_name = path.split('\\')[-1]
#     cv2.imwrite("./calibration_images/saved_images/"+file_name,dst)
