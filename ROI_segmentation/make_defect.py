import cv2 
import numpy as np

#kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], np.uint8) 

kernel = np.ones([56,1], np.uint8) 


for i in range(1, 68):
	for j in range(3, 17):
		img = cv2.imread('./grid_images/'+str(i)+'_'+str(j)+'.bmp')
		dilated_img = np.zeros(img.shape)
		for depth in range (0,3):
			dilated_img[:,:,depth] = cv2.dilate(img[:,:,depth], kernel, iterations=1)
			#dilated_img[:,:,depth] = img[:,:,depth]
			#cv2.imshow(dilated_img)
		if i <= 55:  
			cv2.imwrite('/home/iilab-2080ti-a/Pingu/2020/ROI_segmentation/train/test/images/'+str(i)+'_'+str(j)+'.bmp',dilated_img)
		else:
			cv2.imwrite('/home/iilab-2080ti-a/Pingu/2020/ROI_segmentation/test/test/images/'+str(i)+'_'+str(j)+'.bmp',dilated_img)
