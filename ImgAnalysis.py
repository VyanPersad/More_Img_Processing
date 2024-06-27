import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from Functions.fileFunctions import *
from Functions.ImgAnalysisFunctions import *

filepath = 'CrppdImg_HSV_Set_1'
file = 'IR075_HSV_1_C.png'
image = readFromFile(filepath, file)

grayHistogram = grayHistnoBlk(image)
rgbHistogram = rgbHist(image)

ImgTitles = ['Image','Crppd Gray','Gray Hist','RGB Hist']

f, filmPlot = plt.subplots(1,4, figsize=(10,5))
for i in range (4):
    filmPlot[i].set_title(ImgTitles[i])
    if (i < 2):
        filmPlot[i].imshow(cv2.cvtColor(grayHistogram[i], cv2.COLOR_BGR2RGB))
    elif (i==2):
        filmPlot[i].plot(grayHistogram[2], color='black')
    elif (i==3):
        colorArr = ['r','g','b']
        for j in range(3):
            filmPlot[i].plot(rgbHistogram[1][j], color=colorArr[j])    
 
plt.tight_layout()
plt.show()
