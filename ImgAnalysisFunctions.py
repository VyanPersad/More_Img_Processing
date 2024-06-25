import cv2
import matplotlib.pyplot as plt
import numpy as np

def grayHist(image):
    imgGr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayHist = cv2.calcHist([imgGr],[0],None,[256],[0,256])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imgGr = cv2.cvtColor(imgGr, cv2.COLOR_BGR2RGB)

    return [image, imgGr, grayHist]

def grayHistnoBlk(image):
    msk = (image !=0).astype(np.uint8) 
    #This filters the wholly black pixels.
    #That is those with a value of 0

    imgGr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayHist = cv2.calcHist([imgGr],[0],msk,[256],[0,256])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imgGr = cv2.cvtColor(imgGr, cv2.COLOR_BGR2RGB)

    return [image, imgGr, grayHist]

def rgbHist(image):
    rgbArr = []
    imgCol = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in range(3):
        hist = cv2.calcHist([imgCol],[i],None,[256],[0,256])
        rgbArr.append(hist)
    
    #The second term is an array
    #[0,1,2] = [R,G,B]
    return [imgCol, rgbArr]

def plotrgbHist(rgbArray):
    colorArr = ['r','g','b']
    for i in range(3):
        plt.plot(rgbArray[i], color=colorArr[i])

    plt.tight_layout
    return plt.show()

