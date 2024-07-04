from Functions.ImgAnalysisFunctions import *
from Functions.fileFunctions import *

filepath1 = 'New_Crp_Sample\\Originals'
filepath2 = 'New_Crp_Sample\\Manual_Crp'

ImgTitles = ['Original','New Crop']
ImgOriginal = []
ImgCrp = []

for file in os.listdir(filepath1):
    image = readFromFile_noResize(filepath1, file)
    ImgOriginal.append(image)

for file in os.listdir(filepath2):
    image = readFromFile_noResize(filepath2, file)
    ImgCrp.append(image)

for i in range (5):
    f, filmPlot = plt.subplots(1,len(ImgTitles), figsize=(10,5))
    for j in range(len(ImgTitles)):
        filmPlot[j].set_title(ImgTitles[j])
    
    filmPlot[0].imshow(cv2.cvtColor(ImgOriginal[i], cv2.COLOR_BGR2RGB))
    filmPlot[1].imshow(cv2.cvtColor(ImgCrp[i], cv2.COLOR_BGR2RGB))
 
    plt.tight_layout()
    plt.show()
    plt.close(f)