import os
import cv2
import argparse
import csv
import matplotlib.pyplot as plt

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

def readFromFile(filepath, file):
    #The file param can come from the increment of the directory loop
    # for file in os.listdir(filepath) 
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image file", default=f'{filepath}/{file}')                
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"])    
    image = cv2.resize(image, (300, 300))
    #This image is output as BGR
    return image
    
def writeImgTofile(inputFilename,destFolderPath,fileSuffix,fileType,image):
    makeFolder(destFolderPath)
    #This will isolate the part of the filename 
    #before the .filetype
    #This gives us the basename
    base_name = inputFilename.split(".")[0]
    cv2.imwrite(f'{destFolderPath}/{base_name}_{fileSuffix}.{fileType}',image)

def filmStripPlot(ImgTitles, ImgArray, Num, destFolderPath, file):

    makeFolder(destFolderPath)
    
    base_name = file.split(".")[0]
    #This will only produce a single row of images.
    #The conversion is because cv2 works in BGR
    #plt works in RGB
    f, filmPlot = plt.subplots(1,Num, figsize=(10,5))
    plt.suptitle(f'{base_name}', x=0.05, y=0.9, ha='left', va='top')
    for i in range(Num):
        filmPlot[i].set_title(ImgTitles[i])
        filmPlot[i].imshow(cv2.cvtColor(ImgArray[i], cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.savefig(f'{destFolderPath}/filmStrip_{base_name}.png')
    plt.close(f)

def showfilmStripPlot(ImgTitles, ImgArray, Num, Text, figName='Custom Name'):
    #This will only produce a single row of images.
    #The conversion is because cv2 works in BGR
    #plt works in RGB
    f, filmPlot = plt.subplots(1,Num, figsize=(10,5), num = figName)
    plt.suptitle(f'{figName}', x=0.05, y=0.9, ha='left', va='top')
    plt.figtext(0.1, 0.05, Text, ha='left', va='bottom')
    for i in range(Num):
        filmPlot[i].set_title(ImgTitles[i])
        for j in range(Num):
            filmPlot[j].imshow(cv2.cvtColor(ImgArray[j], cv2.COLOR_BGR2RGB))
 
    plt.tight_layout()
    plt.show()
    plt.close(f)
