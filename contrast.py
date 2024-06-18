import os
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from maskFunctions import *
from fileFunctions import *

for file in os.listdir('Originals/'):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image file", default=f'Originals/{file}')                
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"]) 
    image = cv2.resize(image, (300, 300))
    imgArray = []

    HSV_Contrast = []
    HSV_Contrast.append(image)
    alpha = [1, 1.5, 2.0, 2.5]
    for i in range(4):
        contrasted_img = cv2.convertScaleAbs(image, alpha=alpha[i])
        HSV_Contrast.append(HSVskinMask(contrasted_img))

    img_title = ['Normal','Contrast a = 1','Contrast a = 1.5','Contrast a = 2.0','Contrast a = 2.5']
    
    f, customPlot = plt.subplots(1, 5, figsize=(10,5))
    base_name = file.split(".")[0]

    for i in range(5):
      customPlot[i].set_title(img_title[i])
      for j in range(5):
        customPlot[j].imshow(cv2.cvtColor(HSV_Contrast[j], cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    
    makeFolder('ContrastStrips')
    #plt.show()
    plt.savefig(f'ContrastStrips/ContrastImgStrip_{base_name}.png')
    plt.close(f)