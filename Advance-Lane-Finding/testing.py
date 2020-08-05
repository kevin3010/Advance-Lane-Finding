import numpy as np
import matplotlib as plt
import cv2

import matplotlib.pyplot as plt 
import numpy as np
from glob import glob
import cv2
import random


from utility.transformation import *
from utility.detection import *

def lane_detection(img,display=False):

	histogram = np.sum(img[img.shape[0]//2:],axis=0)
	midpoint = np.int(img.shape[1]/2)

	left_xbase = np.argmax(histogram[:midpoint])
	right_xbase = np.argmax(histogram[midpoint:]) + midpoint

	left_xcurrent = left_xbase
	right_xcurrent = right_xbase

	nwindows = 9 
	margin = 100
	height = img.shape[0]//nwindows

	min_pix = 50

	nonzero = np.nonzero(img)
	nonzeroy = nonzero[0]
	nonzerox = nonzero[1]

	good_left_indices = []
	good_right_indices = []

	output_img = np.zeros_like(img)
	output_img[(img==1)] = 255

	output_img = np.dstack((output_img,output_img,output_img))
	output_img = np.uint8(output_img)

	for window in range(nwindows):

		win_y_low = img.shape[0] - (window+1)*height
		win_y_high = img.shape[0] - (window)*height

		win_xleft_low = left_xcurrent - margin
		win_xleft_high = left_xcurrent + margin

		win_xright_low = right_xcurrent - margin
		win_xright_high = right_xcurrent + margin

		
		# start_pt = (win_xleft_low,win_y_low)
		# end_pt = (win_xleft_high,win_y_high)
		# cv2.rectangle(output_img,start_pt,end_pt,(0,255,0),3)

		# start_pt = (win_xright_low,win_y_low)
		# end_pt = (win_xright_high,win_y_high)
		# cv2.rectangle(output_img,start_pt,end_pt,(0,255,0),3)

		left_window_indices = ((nonzeroy>win_y_low) & (nonzeroy<=win_y_high) \
			& (nonzerox >= win_xleft_low) & (nonzerox <= win_xleft_high)).nonzero()[0]

		right_window_indices = ((nonzeroy>win_y_low) & (nonzeroy<=win_y_high) \
			& (nonzerox >= win_xright_low) & (nonzerox <= win_xright_high)).nonzero()[0]

		if len(left_window_indices) >= min_pix:
			left_xcurrent = np.int(np.mean(nonzerox[left_window_indices]))
		if len(right_window_indices)>= min_pix:
			right_xcurrent = np.int(np.mean(nonzerox[right_window_indices]))

		good_left_indices.append(left_window_indices)
		good_right_indices.append(right_window_indices)


	good_left_indices = np.concatenate(good_left_indices)
	good_right_indices = np.concatenate(good_right_indices)

	leftx = nonzerox[good_left_indices]
	lefty = nonzeroy[good_left_indices]

	rightx = nonzerox[good_right_indices]
	righty = nonzeroy[good_right_indices]

	# output_img[lefty,leftx] = [255,0,0]
	# output_img[righty,rightx] = [0,0,255]

	if display:
		fig,ax = plt.subplots(2,1)
		ax = ax.flatten()

		ax[0].imshow(output_img)
		ax[0].set(title='original')

		ax[1].plot(histogram)

		plt.show()

	return leftx,lefty,rightx,righty,output_img

def prior_lane_detection(img,left_fit,right_fit,display=False):
	
	if len(left_fit)==0 or len(right_fit)==0:
		leftx,lefty,rightx,righty,output_img = lane_detection(img,display=False)
		return leftx,lefty,rightx,righty,output_img

	output_img = np.zeros_like(img)
	output_img[(img==1)] = 255

	output_img = np.dstack((output_img,output_img,output_img))
	output_img = np.uint8(output_img)

	margin = 100

	ploty = np.linspace(0,img.shape[0]-1,img.shape[1])

	nonzero = np.nonzero(img)
	nonzerox = nonzero[1]
	nonzeroy = nonzero[0]

	left_lane_indices = ((nonzerox>(left_fit[0]*nonzeroy**2+left_fit[1]*nonzeroy+left_fit[2]-margin))\
		& (nonzerox<(left_fit[0]*nonzeroy**2+left_fit[1]*nonzeroy+left_fit[2]+margin))).nonzero()[0]

	right_lane_indices = ((nonzerox>(right_fit[0]*nonzeroy**2+right_fit[1]*nonzeroy+right_fit[2]-margin))\
		& (nonzerox<(right_fit[0]*nonzeroy**2+right_fit[1]*nonzeroy+right_fit[2]+margin))).nonzero()[0]

	leftx = nonzerox[left_lane_indices]
	lefty = nonzeroy[left_lane_indices]

	rightx = nonzerox[right_lane_indices]
	righty = nonzeroy[right_lane_indices]

	return leftx,lefty,rightx,righty,output_img





def fit_polynomial(img,left_fit,right_fit,display=False):

	leftx,lefty,rightx,righty,output_img = prior_lane_detection(img,left_fit,right_fit,display=display)
	leftx,lefty,rightx,righty,output_img = lane_detection(img,display=display)


	left_fit = np.polyfit(lefty,leftx,2)
	right_fit = np.polyfit(righty,rightx,2)

	ploty = np.linspace(0,img.shape[0]-1,img.shape[0])

	try:
		left_fitx = left_fit[0]*(ploty**2) + left_fit[1]*(ploty) + left_fit[2]
		right_fitx = right_fit[0]*(ploty**2) + right_fit[1]*(ploty) + right_fit[2] 	
	except:
		print("failed to print line")
		left_fitx = 1*(ploty**2) + 1*(ploty)
		right_fitx = 1*(ploty**2) + 1*(ploty)
		pass

	

	output_img[lefty,leftx] = [255,0,0]
	output_img[righty,rightx] = [0,0,255]

	copy = np.zeros_like(output_img)
	copy = copy + 255

	left_pts = np.int32(np.flipud(np.transpose(np.vstack([left_fitx,ploty]))))
	right_pts = np.int32(np.transpose(np.vstack([right_fitx,ploty])))
	pts = np.vstack([left_pts,right_pts])

	copy = cv2.polylines(output_img,[left_pts,right_pts],False,(0,0,255),4)
	# copy = cv2.fillPoly(copy,[pts],(0,255,0))


	if display:
		plt.plot(left_fitx,ploty,color='yellow')
		plt.plot(right_fitx,ploty,color='yellow')
		plt.imshow(copy)
		plt.show()

	return left_pts,right_pts,left_fit,right_fit,output_img


cap = cv2.VideoCapture('./images/videos/project_video.mp4')
left_fit = []
right_fit = []
while(cap.isOpened()):


	ret, frame = cap.read()
	img = frame

	Minv,warped = perspective_transform(img,display = False)

	combined_img = combined_binary(warped,display=False)
	left_pts,right_pts,left_fit,right_fit,detect_img = fit_polynomial(combined_img,left_fit,right_fit,display=False)

	final = merge(img,warped,detect_img,left_pts,right_pts,Minv,display=False)
	
	cv2.imshow('frame',final)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()




# paths = glob('./images/test*.jpg')
# paths = [paths[4]]
# for path in paths:

#   img = cv2.imread(path)
#   img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#   Minv,warped = perspective_transform(img,display = False) 

#   combined_img = combined_binary(warped,display=False)

#   left_fit = (2.0288e-4,-1.979e-1,3.5912e+2)
#   right_fit = (5.5198e-4,-5.7522e-1,1.1628e+3)

#   left_fit = (6.0288e-4,-1.979e-1,3.5912e+2)
#   right_fit = (4.5198e-4,-0.7522e-1,1.1628e+3)

#   left_pts,right_pts,left_fit,right_fit,detect_img = \
#   fit_polynomial(combined_img,left_fit,right_fit,display=True)

  # final = merge(img,warped,detect_img,left_pts,right_pts,Minv,display=False)