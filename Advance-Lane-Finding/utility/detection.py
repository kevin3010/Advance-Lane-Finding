import numpy as np
import cv2
import matplotlib.pyplot as plt

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

def fit_polynomial(img,display=False):

	leftx,lefty,rightx,righty,output_img = lane_detection(img,display=False)

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
	# copy = copy + 255

	left_pts = np.int32(np.flipud(np.transpose(np.vstack([left_fitx,ploty]))))
	right_pts = np.int32(np.transpose(np.vstack([right_fitx,ploty])))
	pts = np.vstack([left_pts,right_pts])

	copy = cv2.polylines(copy,[left_pts,right_pts],False,(0,0,255),40)
	copy = cv2.fillPoly(copy,[pts],(0,255,0))


	if display:
		plt.plot(left_fitx,ploty,color='yellow')
		plt.plot(right_fitx,ploty,color='yellow')
		plt.imshow(copy)
		plt.show()

	return left_pts,right_pts,output_img

def merge(original,warped,detect_img,left_pts,right_pts,Minv,display=False):



	shape = (original.shape[1],original.shape[0])

	pts = np.vstack([left_pts,right_pts])
	warped = cv2.polylines(warped,[left_pts,right_pts],False,(0,0,255),40)
	warped = cv2.fillPoly(warped,[pts],(0,255,0))

	detect_img = cv2.polylines(detect_img,[left_pts,right_pts],False,(255,255,0),10)

	warped = cv2.warpPerspective(warped, Minv, shape, flags=cv2.INTER_LINEAR)

	final = cv2.addWeighted(original,1.0,warped,0.3,0)

	detect_img = cv2.resize(detect_img,(318,180))

	hspace,vspace = 20,20
	height,width = detect_img.shape[0],detect_img.shape[1]

	final[hspace:hspace+height,vspace:vspace+width] = detect_img

	if display:
		plt.imshow(final)
		plt.show()

	return final
    	
