import os
import cv2
import argparse
import csv
import matplotlib.pylab as plt

def write_to_csv(output_file_path, fieldnames, data):
    
    file_exists = os.path.exists(output_file_path)
    with open(output_file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)  
        # Write header if the file is newly created
        if not file_exists:
            writer.writeheader()
        # Write rows
        for row in data:
            writer.writerow(row)
  
def makeFolder(folderpath):
    if os.path.exists(folderpath):
        return
        #print("Path Exists")
    else:
        os.makedirs(folderpath)
        print("Path does not exist creating....")

def readFromFile(filepath):
    for file in os.listdir(filepath):
        # construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", help="path to the image file", default=f'{filepath}]/{file}')                
        args = vars(ap.parse_args())
        image = cv2.imread(args["image"])    
        image = cv2.resize(image, (300, 300))

        return image
    
def writeImgTofile(inputFilename,destFolderPath,fileSuffix,fileType,image):
    makeFolder(destFolderPath)
    #This will isolate the part of the filename 
    #before the .filetype
    #This gives us the basename
    base_name = inputFilename.split(".")[0]
    cv2.imwrite(f'{destFolderPath}/{base_name}_{fileSuffix}.{fileType}',image)
