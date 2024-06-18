import os
import argparse
import cv2
import matplotlib.pyplot as plt

for file in os.listdir('Originals/'):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image file", default=f'Originals/{file}')                
    args = vars(ap.parse_args())
    
    image = cv2.imread(args["image"]) 
    image = cv2.resize(image, (300, 300))
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.imshow('Test',image)
    cv2.waitKey(0)
    
    plt.imshow(image2)
    plt.show()

