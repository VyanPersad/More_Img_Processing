import os
import numpy as np
import argparse
import cv2

fcount = 0

for file in os.listdir('Originals/'):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image file", default=f'Originals/{file}')
                    
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])    
    image = cv2.resize(image, (300, 300))

    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    #Shadow Mask
    converted_c = cv2.cvtColor(converted, cv2.COLOR_BGR2HSV)
    lshad = np.array([0, 0, 0], dtype="uint8")
    ushad = np.array([180, 50, 15], dtype="uint8")
    shadow_mask = cv2.inRange(converted_c, lshad, ushad)
    #This should show the shadowed regionwith a green tint.
    img_wo_shadow = cv2.bitwise_and(converted, converted, mask=~shadow_mask)
    
    base_name = file.split(".")[0]
    #This specifically writes the image to a file called skin1.png
    cv2.imwrite(f'CroppedImgs/{base_name}_C.png',img_wo_shadow)
