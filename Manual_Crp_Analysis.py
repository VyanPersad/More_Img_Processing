import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from Functions.fileFunctions import *
from Functions.ImgAnalysisFunctions import *

filepath = 'NewCropped'
#file = 'IR017.jpg'
destFolderPath = 'OutputFolder/Manual_N_Crp_Analysis'
filePrefix = 'new_Crp'

for file in os.listdir(filepath):
    image = readFromFile(filepath, file)

    grayHistogram = grayHistnoBlk(image)
    rgbHistogram = rgbHist(image)

    ImgTitles = ['Image','Gray Hist','RGB Hist']

    f, filmPlot = plt.subplots(1,len(ImgTitles), figsize=(10,5))
    base_name = file.split(".")[0]
    plt.suptitle(f'{base_name}', x=0.05, y=0.9, ha='left', va='top')
    #plt.figtext(0.1, 0.05, Text, ha='left', va='bottom')
    for i in range (len(ImgTitles)):
        filmPlot[i].set_title(ImgTitles[i])
        if (i < 1):
            filmPlot[i].imshow(cv2.cvtColor(grayHistogram[i], cv2.COLOR_BGR2RGB))
        elif (i==1):
            filmPlot[i].plot(grayHistogram[2], color='black')
        elif (i==2):
            colorArr = ['r','g','b']
            for j in range(3):
                filmPlot[i].plot(rgbHistogram[1][j], color=colorArr[j])    
    
    plt.tight_layout()
    makeFolder(destFolderPath)
    plt.savefig(f'{destFolderPath}/{filePrefix}_{base_name}.png')
    #plt.show()
    plt.close(f)