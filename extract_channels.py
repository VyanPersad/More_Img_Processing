import os
import cv2
import numpy as np
import csv
import random
import math
from statistics import mean
from Functions.fileFunctions import write_to_csv
from Functions.fileFunctions import readFromFile
from Functions.ImgAnalysisFunctions import k_means

RGB_SCALE = 255
CMYK_SCALE = 100

def sRGBtoLinearRGB(c):
    if c <= 0.04045:
        return c / 12.92
    else:
        return round((((c + 0.055) / 1.055) ** 2.4),2)

def rgb_to_cmyk(r, g, b):
    if (r, g, b) == (0, 0, 0):
        # black
        return 0, 0, 0, CMYK_SCALE

    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / RGB_SCALE
    m = 1 - g / RGB_SCALE
    y = 1 - b / RGB_SCALE

    # extract out k [0, 1]
    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy

    # rescale to the range [0,CMYK_SCALE]
    return int(c * CMYK_SCALE), int(m * CMYK_SCALE), int(y * CMYK_SCALE), int(k * CMYK_SCALE)

def rgbToLab(r, g, b) :
    r = r / 255
    g = g / 255
    b = b / 255

    if r > 0.04045:
        r = (r + 0.055) / 1.055 ** 2.4
    else:
        r = r / 12.92

    if g > 0.04045:
        g = (g + 0.055) / 1.055 ** 2.4
    else:
        g = g / 12.92

    if b > 0.04045:
        b = (b + 0.055) / 1.055 ** 2.4
    else:
        b = b / 12.92

    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

    if x > 0.008856:
        x = x ** (1 / 3)
    else:
        x = (7.787 * x) + 16 / 116
    if y > 0.008856:
        y = y ** (1 / 3)
    else:
        y = (7.787 * y) + 16 / 116
    if z > 0.008856:
        z = z ** (1 / 3)
    else:
        z = (7.787 * z) + 16 / 116

    return round(((116 * y) - 16),3), round((500 * (x - y)),3), round((200 * (y - z)),3)

def rgbToHsv(r, g, b):
    rabs = r / 255
    gabs = g / 255
    babs = b / 255
    v = max(rabs, gabs, babs)
    diff = v - min(rabs, gabs, babs)
    diffc = lambda c: (v - c) / 6 / diff + 1 / 2
    percentRoundFn = lambda num: round(num * 100) / 100
    if diff == 0:
        h = s = 0
    else:
        s = diff / v
        rr = diffc(rabs)
        gg = diffc(gabs)
        bb = diffc(babs)
        if rabs == v:
            h = bb - gg
        elif gabs == v:
            h = (1 / 3) + rr - bb
        elif babs == v:
            h = (2 / 3) + gg - rr
        if h < 0:
            h += 1
        elif h > 1:
            h -= 1
    return round(h * 360),percentRoundFn(s * 100),percentRoundFn(v * 100)

def rgbToLuminance(r, g, b):
      return round((((0.2126*r/255)+(0.7152*g/255)+(0.0722*b/255))*100),2)

def temperature2rgb(kelvin):
    temperature = kelvin / 100.0
    if temperature < 66.0:
        red = 255
    else:
        red = temperature - 55.0
        red = 351.97690566805693 + 0.114206453784165 * red - 40.25366309332127 * math.log(red)
        if red < 0:
            red = 0
        if red > 255:
            red = 255
    if temperature < 66.0:
        green = temperature - 2
        green = -155.25485562709179 - 0.44596950469579133 * green + 104.49216199393888 * math.log(green)
        if green < 0:
            green = 0
        if green > 255:
            green = 255
    else:
        green = temperature - 50.0
        green = 325.4494125711974 + 0.07943456536662342 * green - 28.0852963507957 * math.log(green)
        if green < 0:
            green = 0
        if green > 255:
            green = 255
    if temperature >= 66.0:
        blue = 255
    else:
        if temperature <= 20.0:
            blue = 0
        else:
            blue = temperature - 10
            blue = -254.76935184120902 + 0.8274096064007395 * blue + 115.67994401066147 * math.log(blue)
            if blue < 0:
                blue = 0
            if blue > 255:
                blue = 255
    return {"red": round(red), "blue": round(blue), "green": round(green)}

def rgbToTemperature(r, g, b):
    epsilon = 0.4
    minTemperature = 1000
    maxTemperature = 40000
    while maxTemperature - minTemperature > epsilon:
        temperature = (maxTemperature + minTemperature) / 2
        testRGB = temperature2rgb(temperature)
        if (testRGB["blue"] / testRGB["red"]) >= (b / r):
            maxTemperature = temperature
        else:
            minTemperature = temperature
    return round(temperature)

def rgbToRyb(r, g, b):
    # Remove the whiteness from the color.
    w = min(r, g, b)
    r -= w
    g -= w
    b -= w

    mg = max(r, g, b)

    # Get the yellow out of the red+green.
    y = min(r, g)
    r -= y
    g -= y

    # If this unfortunate conversion combines blue and green, then cut each in
    # half to preserve the value's maximum range.
    if b and g:
        b /= 2.0
        g /= 2.0

    # Redistribute the remaining green.
    y += g
    b += g

    # Normalize to values.
    my = max(r, y, b)
    if my:
        n = mg / my
        r *= n
        y *= n
        b *= n

    # Add the white back in.
    r += w
    y += w
    b += w

    # And return back the ryb typed accordingly.
    return int(r), int(y), int(b)

def rgbToXyz(r, g, b):
    r = sRGBtoLinearRGB(r / 255)
    g = sRGBtoLinearRGB(g / 255)
    b = sRGBtoLinearRGB(b / 255)

    X = 0.4124 * r + 0.3576 * g + 0.1805 * b
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = 0.0193 * r + 0.1192 * g + 0.9505 * b

    return int(X * 100), int(Y * 100), int(Z * 100)

def getRandoPxs(file_path, num_pxs=2):
    img = Image.open(file_path)
    width, height = img.size

    def rejectblk_whit(pxl):
        r,g,b = pxl
        return r==g==b ==0 or r==b==g==255
    
    selected_pxls=[]
    while len(selected_pxls)<num_pxs:
        x,y = random.randint(0,width-1), random.randint(0,height-1)
        pxl_val = img.getpixel((x,y))
        if not rejectblk_whit(pxl_val):
            selected_pxls.append(pxl_val)

    return selected_pxls

def calc_bright(pixel):
    r,g,b = pixel
    return 0.299*r+0.587*g+0.114*b

def light_dark(pixel_list):
    normal = None
    hyper = None
    max_lum = float("-inf")
    min_lum = float("inf")
    for pxl in pixel_list:
        lum = calc_bright(pxl)
        if lum > max_lum:
            max_lum = lum
            normal = pxl
        if lum < min_lum:
            min_lum = lum
            hyper = pxl

    return normal, hyper        

def avg_colour(img):
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    return avg_color

def rounded(r,g,b):
    rr=round(r,2)
    rg=round(g,2)
    rb=round(b,2)

    return rr,rg,rb

def rgb2(function, r1,r2,r3, dataArray):
    dataArray.append(function(r1,r2,r3))
    return
#0<--Blk+++White-->255

hcmyk = ""
ncmyk = ""
hlab = ""
nlab = ""
hhsv = ""
nhsv = ""
hlum = ""
nlum = ""
htemp = ""
ntemp = ""
hryb = ""
nryb = ""
hxyz = ""
nxyz = ""

filepath = 'Manual_Crop'

for file in os.listdir(f'{filepath}'):    
    imagebgr = readFromFile(filepath, file)   
    segmented_crppd, center = k_means(imagebgr, 2)
    centArr = []
    # Centre Details
    for i in range(3):
        if (mean([center[i][0], center[i][1], center[i][2]]) > 10):
            centArr.append([center[i][0], center[i][1], center[i][2]])

    if (mean([centArr[0][0], centArr[0][1], centArr[0][2]])<mean([centArr[1][0], centArr[1][1], centArr[1][2]])):  
        # 0 = Hyper
        # 1 = Normal  
        a = 0

    elif(mean([centArr[0][0], centArr[0][1], centArr[0][2]])>mean([centArr[1][0], centArr[1][1], centArr[1][2]])):
        # 0 = Normal
        # 1 = Hyper
        a = 1

    hyperData = []
    normalData = []
    if (a == 0):
        j = 0
        k = 1
    elif (a == 1):
        j = 1
        k = 0

    hcmyk = rgb_to_cmyk(centArr[j][2],centArr[j][1],centArr[j][0])
    hlab = rgbToLab(centArr[j][2],centArr[j][1],centArr[j][0])
    hhsv = rgbToHsv(centArr[j][2],centArr[j][1],centArr[j][0])
    hlum = rgbToLuminance(centArr[j][2],centArr[j][1],centArr[j][0])
    htemp = rgbToTemperature(centArr[j][2],centArr[j][1],centArr[j][0])
    hryb = rgbToRyb(centArr[j][2],centArr[j][1],centArr[j][0])
    hxyz = rgbToXyz(centArr[j][2],centArr[j][1],centArr[j][0])

    ncmyk = rgb_to_cmyk(centArr[k][2],centArr[k][1],centArr[k][0])
    nlab = rgbToLab(centArr[k][2],centArr[k][1],centArr[k][0])
    nhsv = rgbToHsv(centArr[k][2],centArr[k][1],centArr[k][0])
    nlum = rgbToLuminance(centArr[k][2],centArr[k][1],centArr[k][0])
    ntemp = rgbToTemperature(centArr[k][2],centArr[k][1],centArr[k][0])
    nryb = rgbToRyb(centArr[k][2],centArr[k][1],centArr[k][0])
    nxyz = rgbToXyz(centArr[k][2],centArr[k][1],centArr[k][0])

  
    allData = [{'HCMYK': hcmyk, 'NCMYK': ncmyk, 'HLAB': hlab, 'NLAB': nlab, 'HHSV':hhsv, 'NHSV':nhsv, 
                'HLUM':hlum, 'NLUM':nlum, 'HTEMP':htemp, 'NTEMP':ntemp, 'HRYB': hryb, 'NRYB': nryb, 'HXYZ': hxyz, 'NXYZ':nxyz}]

    hyperData = [{'HCMYK': hcmyk, 'HLAB': hlab,'HHSV':hhsv, 'HLUM':hlum,'HTEMP':htemp, 'HRYB': hryb, 'HXYZ': hxyz}]
    normalData = [{'NCMYK': ncmyk,'NLAB': nlab,'NHSV':nhsv, 'NLUM':nlum,'NTEMP':ntemp, 'NRYB': nryb, 'NXYZ':nxyz}]
    
    header_names = ['HCMYK', 'NCMYK', 'HLAB', 'NLAB', 'HHSV', 'NHSV', 'HLUM', 'NLUM', 'HTEMP', 'NTEMP', 'HRYB', 'NRYB', 'HXYZ','NXYZ']
    hyper_names = ['HCMYK','HLAB','HHSV', 'HLUM','HTEMP','HRYB','HXYZ',]
    normal_names = ['NCMYK','NLAB','NHSV', 'NLUM','NTEMP','NRYB','NXYZ']
    
    write_to_csv('hyperData_channels_frm_ Man_Crp.csv', hyper_names, hyperData)
    write_to_csv('normalData_channels_frm_ Man_Crp.csv', normal_names, normalData)
    
   