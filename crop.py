import os
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from maskFunctions import *
from fileFunctions import *

f, compPlot = plt.subplots(2, 2)
hsvCrp = []
ycbcrCrp = []


for file in os.listdir('Originals/'):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image file", default=f'Originals/{file}')                
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"]) 
    image = cv2.resize(image, (300, 300))

    crppd_img_HSV_1 = HSVskinMask(image,[20,255,255],[3,15,10])
    crppd_img_HSV_2 = HSVskinMask(image,[25,255,255],[0,40,0])

    crppd_img_YCb = YCbCrskinMask(image,[255,173,133],[0,138,67])
    #This way only the 4 imgs for this one image is stored. 
    imgArray = []
    imgArray.append(image)
    imgArray.append(crppd_img_HSV_1)
    imgArray.append(crppd_img_HSV_2)
    imgArray.append(crppd_img_YCb)

    #hsvCrp.append(crppd_img_HSV)
    #ycbcrCrp.append(crppd_img_YCb)

    writeImgTofile(file, 'CrppdImg_HSV_Set_1', 'HSV_1_C', 'png', crppd_img_HSV_1)
    writeImgTofile(file, 'CrppdImg_HSV_Set_2', 'HSV_2_C', 'png', crppd_img_HSV_2)

    writeImgTofile(file, 'CrppdImg_YCrCb', 'YCrCb_C', 'png', crppd_img_YCb)

    f, customPlot = plt.subplots(1, 4, figsize=(15,5))
    img_title = ['Normal','HSV_1','HSV_2','YCrCb']
    base_name = file.split(".")[0]


    for i in range(4):
      customPlot[i].set_title(img_title[i])
      #cv2 works in BGR plt works in RGB hence the conversion 
      for j in range(4):
        customPlot[j].imshow(cv2.cvtColor(imgArray[j], cv2.COLOR_BGR2RGB))
      
    plt.tight_layout()
    
    makeFolder('ImageStrips')
    #plt.show()
    plt.savefig(f'ImageStrips/ImgStrip_{base_name}.png')
    plt.close(f)
