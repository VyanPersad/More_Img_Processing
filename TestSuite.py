from Functions.ImgAnalysisFunctions import *
from Functions.fileFunctions import *


ImgArr = []

ImgArr.append(readFromFile_noResize('New_Crp_Sample\\Originals','IR005.jpg'))
ImgArr.append(readFromFile_noResize('New_Crp_Sample','IR005-Op1.jpg'))
ImgArr.append(readFromFile_noResize('New_Crp_Sample','IR005-Op2.jpg'))

ImgTitles = ['Original', 'Opt. 1','Opt. 2']

showfilmStripPlot(ImgTitles, ImgArr, None, 'IR005')
