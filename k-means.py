import os
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from fileFunctions import *
from maskFunctions import HSVskinMask

def k_means(image):
    pxl_val = image.reshape((-1,3))
    pxl_val = np.float32(pxl_val)
    #print(pxl_val.shape)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3
    _, labels, (centers) = cv2.kmeans(pxl_val, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image, centers

imgArray = []   
centerArr = []

filepath = "Originals/IR115.jpg"

#for file in os.listdir(filepath):
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", help="path to the image file", default=f'{filepath}]/{file}')                
#args = vars(ap.parse_args())
#image = cv2.imread(args["image"])    
image = cv2.imread(f'{filepath}')
image = cv2.resize(image, (300, 300))
normal = image
imagebgr = normal

crppd_img_HSV_1 = HSVskinMask(imagebgr,[20,255,255],[3,15,10])

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
segmented_image, centre = k_means(image)
segmented_crppd, _ = k_means(crppd_img_HSV_1)
'''
# show the image
f, customPlot = plt.subplots(1, 4, figsize=(10,5))
img_title = ['Normal','Segmented','Cropped','Segmented Cropped']
cluster_title = ['1','2','3']

imgArray.append(cv2.cvtColor(normal, cv2.COLOR_BGR2RGB))
imgArray.append(segmented_image)
imgArray.append(cv2.cvtColor(crppd_img_HSV_1, cv2.COLOR_BGR2RGB))
imgArray.append(cv2.cvtColor(segmented_crppd, cv2.COLOR_BGR2RGB))

for i in range(4):
    customPlot[i].set_title(img_title[i])
    for j in range(4):
        customPlot[j].imshow(imgArray[j])
'''

image_pyplot = np.zeros((1, 1, 3), dtype=np.uint8)
image_pyplot[0, 0] = (centre[0][0], centre[0][1], centre[0][2]) 
image_pyplot = cv2.cvtColor(image_pyplot, cv2.COLOR_BGR2RGB)
# Display the image
plt.imshow(image_pyplot)
plt.tight_layout()
plt.show()

