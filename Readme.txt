Place the images of the files to be cropped into the Originals folder.
If it does not exist then add it in an Originals folder. 
Ensure that the filenames are in the form IRXXX.jpg this helps in labelling the cropped files and 
navigating the files.

It is important to note that cv2 libraries use BGR as the output so all images coming out of the read function will be in 
that color space. 
Similarly it is best to use BGR when using any of the functions from the cv2 library.

The last stage would be to convert it to RGB for siaply purposes.
The matplot livray from which pyplot and plt is used usses RGB so to diaply the image you can convert to RGB.

See code below:

img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

All the functions work with BGR so you only have to convert to RGB when you want to print or display.

For the functions that output a histogram you need to use the plt.plot() to genrate the image/plot.
For the functions that return multiple values you need to refer to the specific output as you would an array.

someVar - someFunct()

where someFunc returns a, b, c
then 
    someVar[0] = a
    someVar[1] = b
    someVar[2] = c