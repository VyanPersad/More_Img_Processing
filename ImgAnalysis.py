import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from Functions.fileFunctions import *
from Functions.ImgAnalysisFunctions import *

def plot2Img(graphPlot):
    plt.tight_layout()
    plt.plot(graphPlot)
    plt.savefig(graphPlot)
    plt.show()
    graphPlotImg = cv2.imread(f'{graphPlot}.png')

    return graphPlotImg

filepath = 'CrppdImg_HSV_Set_1'
file = 'IR009_HSV_1_C.png'
image = readFromFile(filepath, file)

grayHistogram = grayHistnoBlk(image)

Num = 3
ImgTitles = ['Image','Crppd Gray','Gray Hist']

img_array = []
img_array.append(image)
img_array.append(grayHistogram[1])

histo = grayHistogram[2]
'''
plt.tight_layout
plt.plot(histo)
plt.savefig('histo.png')
plt.show()
histImg = cv2.imread('histo.png')
'''
histImg = plot2Img(histo)

img_array.append(histImg)


showfilmStripPlot(ImgTitles, img_array,3)




