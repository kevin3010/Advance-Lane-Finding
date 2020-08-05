import matplotlib.pyplot as plt 
import numpy as np
from glob import glob
import cv2
import random

from utility.transformation import *
from utility.detection import *
from testing import fit_polynomial

cap = cv2.VideoCapture('./images/project_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

out = cv2.VideoWriter('filename.avi',cv2.VideoWriter_fourcc(*'XVID'),10, (640,480)) 

while(cap.isOpened()):
    ret, frame = cap.read()
    # img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    img = frame

    Minv,warped = perspective_transform(img,display = False)

    combined_img = combined_binary(warped,display=False)
    try: 
    	left_pts,right_pts,detect_img = fit_polynomial(combined_img,display=False)
    	final = merge(img,warped,detect_img,left_pts,right_pts,Minv,display=False)
    	out.write(final)

    except:
	    cv2.imshow('frame',final)
	    if cv2.waitKey(100) & 0xFF == ord('q'):
	        break

print("Done")
out.release()
cap.release()
cv2.destroyAllWindows()
# paths = glob('./images/test*.jpg')
# paths = [paths[4]]
# for path in paths:

#   img = cv2.imread(path)
#   img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#   Minv,warped = perspective_transform(img,display = False) 

#   combined_img = combined_binary(warped,display=False)

#   left_pts,right_pts,detect_img = fit_polynomial(combined_img,None,display=False)

#   final = merge(img,warped,detect_img,left_pts,right_pts,Minv,display=True)


    




