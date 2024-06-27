import os
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from Functions.fileFunctions import *
from Functions.maskFunctions import HSVskinMask
from Functions.ImgAnalysisFunctions import k_means

imgArray = []   
centerArr = []
img_title = ['Image','Cropped','k-Means','Hyper','Normal']

filepath = 'Originals'
file = 'IR080.jpg'
imagebgr = readFromFile(filepath, file)

crppd_img_HSV_1 = HSVskinMask(imagebgr,[20,255,255],[3,15,10])
segmented_crppd, centre = k_means(crppd_img_HSV_1)

#Builds the image array.
imgArray.append(imagebgr)
imgArray.append(crppd_img_HSV_1)
imgArray.append(segmented_crppd)
for i in range(2):
    image_pyplot = np.zeros((1, 1, 3), dtype=np.uint8)
    image_pyplot[0, 0] = (centre[i][0], centre[i][1], centre[i][2]) 
    imgArray.append(image_pyplot)

# Display the image
showfilmStripPlot(img_title, imgArray, 5)




