import os
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from Functions.fileFunctions import *
from Functions.maskFunctions import HSVskinMask
from Functions.ImgAnalysisFunctions import k_means

  
centerArr = []
img_title = ['Image','Cropped','k-Means','C1','C2','C3']

filepath = 'Originals'
file = 'IR005.jpg'

#for file in os.listdir(filepath):
imagebgr = readFromFile(filepath, file)

crppd_img_HSV_1 = HSVskinMask(imagebgr,[20,255,255],[3,15,10])
segmented_crppd, centre = k_means(crppd_img_HSV_1)
imgArray = [] 
#Builds the image array.
imgArray.append(imagebgr)
imgArray.append(crppd_img_HSV_1)
imgArray.append(segmented_crppd)
for i in range(3):
    image_pyplot = np.zeros((1, 1, 3), dtype=np.uint8)
    image_pyplot[0, 0] = (centre[i][0], centre[i][1], centre[i][2]) 
    imgArray.append(image_pyplot)

# Centre Details
centreDeets = f' C1-BGR- {centre[0][0], centre[0][1], centre[0][2]} \
\n C2-BGR- {centre[1][0], centre[1][1], centre[1][2]} \
\n C3-BGR- {centre[2][0], centre[2][1], centre[2][2]} \
    '
# Display the image
#
showfilmStripPlot(img_title, imgArray, 6, centreDeets, f'{file.split(".")[0]}')
#filmStripPlot(img_title,imgArray,6,'k-means_filmStrips',file)



