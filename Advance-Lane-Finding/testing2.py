from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import cv2

# a = np.array([[[10,20,30],[123,12,34]],[[10,4,23],[34,5,56]],[[10,4,23],[34,5,56]]])
# a = a.reshape((-1,3))
# print(a)

# Read a color image
img = cv2.imread("./images/test/53.png")

# Select a small fraction of pixels to plot by subsampling it
scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)

# Convert subsampled image to desired color space(s)
img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting
img_small_hsv = img_small_HSV / 255.

ax = plt.axes(projection='3d')
ax.set_xlabel('R', fontsize=16, labelpad=16)
ax.set_ylabel('G', fontsize=16, labelpad=16)
ax.set_zlabel('B', fontsize=16, labelpad=16)

R = img_small_HSV[:,:,0]
G = img_small_HSV[:,:,1]
B = img_small_HSV[:,:,2]

ax.scatter3D(R,G,B,c=img_small_rgb.reshape(-1,3),)



plt.show()

