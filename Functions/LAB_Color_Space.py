import os
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from Functions.maskFunctions import *
from Functions.fileFunctions import *

filepath = 'Originals'
destFilePath = ''

def LAB_Color_Space(filepath, destFolderPath, saveState):
  for file in os.listdir(filepath):

      image = readFromFile(filepath, file)
      image_LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

      LAB_Contrast = []
      LAB_Contrast.append(image)
      alpha = [1, 1.5, 2.0, 2.5]
      
      for i in range(4):
          lab_img = cv2.convertScaleAbs(image_LAB, alpha[i])
          LAB_Contrast.append(lab_img)

      img_title = ['Normal','Contrast a = 1','Contrast a = 1.5','Contrast a = 2.0','Contrast a = 2.5']
      
      f, customPlot = plt.subplots(1, 5, figsize=(10,5))
      base_name = file.split(".")[0]

      for i in range(5):
        customPlot[i].set_title(img_title[i])
        for j in range(5):
          customPlot[j].imshow(cv2.cvtColor(LAB_Contrast[j], cv2.COLOR_BGR2RGB))

      plt.tight_layout()
      
      makeFolder(destFolderPath)
      if (saveState == 0):
        plt.show()
      elif (saveState == 1):  
        plt.savefig(f'{destFolderPath}/LABImgStrip_{base_name}.png')
      
      plt.close(f)