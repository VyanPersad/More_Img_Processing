from Functions.ImgAnalysisFunctions import *
from Functions.fileFunctions import *

folderpath = 'OutputFolder\\Cropped_Sets\\CrppdImg_HSV_Set_1'
destFolderPath = 'OutputFolder\\Original_Crp_Analysis'
#file = 'pic1.jpg'

for file in os.listdir(folderpath):
    imgAnalysisFile1x4(folderpath,file, 1, destFolderPath)
