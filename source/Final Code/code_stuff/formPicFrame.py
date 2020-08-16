import os
import numpy as np
import cv2

def formPicFrame(folder, read_from):
    i=0
    f_image=open(read_from + "/picdata.txt",'wb')
    f_image1=open(read_from + "/picdata_val.txt",'wb')
    f_image2=open(read_from + "/picdata_test.txt",'wb')
    os.chdir(folder)
    for file in os.listdir(folder):
      
#        if file[0]!='0':  #I have an issue with 'Thumbs.db'
#            continue
#        i+=1
            
        #Open and process image
        img = cv2.imread(file,0)
        y=np.asarray(img)
        
        image=y.ravel()  #Transform 2D matrix into 1D array
        image.flags.writeable = True
        # Adjust values to be between 0 and 255 only
        image[image < 100] = 0
        image[image >= 100] = 255
        
        if i <= 6:
            f_image.write(image)
        elif i <= 8:
            f_image1.write(image)
        else:
            f_image2.write(image)
            i = 0
            
    f_image.close()
    f_image1.close()
    f_image2.close()