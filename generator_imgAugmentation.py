from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os,subprocess
import numpy as np
from skimage.transform import resize
from skimage import data
from scipy.misc import imresize
import random
from PIL import Image
'''
Script makes different perspectives of an image. Crops it to 100x100 depending on which side is bigger.
'''
#set parameters for the ImageGenerator
datagen = ImageDataGenerator(
        samplewise_center=False,
        samplewise_std_normalization=False,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=0.9 ,
        fill_mode='nearest')


def imcrop_tosquare(img):
    if img.size[0] > img.size[1]: #width > height
        extra = (img.size[0] - img.size[1])
        #centre cropping 
        if extra % 2 == 0:
            crop1 = img.crop((extra // 2,0,img.size[0]-extra // 2,img.size[1]))
        else:
            crop1 = img.crop((extra // 2 + 1,0,img.size[0]-extra // 2,img.size[1]))
        #right side cropping
        crop2 = img.crop((0,0,img.size[0] - extra, img.size[1]))
        #left side cropping
        crop3 = img.crop((extra,0,img.size[0],img.size[1]))

    elif img.size[1] > img.size[0]: #width < height
        extra = (img.size[1] - img.size[0])
        #centre cropping
        if extra % 2 == 0:
            crop1 = img.crop((0,extra // 2,img.size[0],img.size[1]-extra // 2))
        else:
            crop1 = img.crop((0,extra // 2 + 1,img.size[0],img.size[1]-extra // 2))
        #top cropping
        crop2 = img.crop((0,extra,img.size[0],img.size[1]))
        #bottom cropping
        crop3 = img.crop((0,0,img.size[0],img.size[1]-extra))
    else:                            #width == height
        #centre cropping only
        crop1 = img
        crop2 = img
        crop3 = img

    crop1 = crop1.resize((100,100),Image.ANTIALIAS)
    crop2 = crop2.resize((100,100),Image.ANTIALIAS)
    crop3 = crop3.resize((100,100),Image.ANTIALIAS)

    return crop1, crop2, crop3
 

dirname ="/home/aakash/Desktop/genImage/Vehicle Classify/" # specify images where the original images are kept
outputDirname = "/home/aakash/Desktop/genImage/Vehicle Classify Augmented/" # specify where to save augmented images
j = 0
folderNameList = os.listdir(dirname)
print(folderNameList)
for folderName in folderNameList:
    imageList = os.listdir(dirname+folderName+"/")

    if os.path.isdir(outputDirname+folderName) == False:
        os.mkdir(outputDirname+folderName+"/")

    preNum = 0
    for image in imageList:
        #print("/home/aakash/Desktop/genImage/Vehicle Classify/"+folderName+"/"+image)
        try:
            img = load_img(dirname+folderName+"/"+image)
        except Exception as e:
            continue
        
        width, height = img.size

        if width > 100 and height > 100:
            (img1,img2,img3) = imcrop_tosquare(img)
            #img1.save("centre.jpg")
            #img2.save("side.jpg")
            #img3.save("tops.jpg")
            for img in (img1, img2, img3):
                x = img_to_array(img)  
                x = x.reshape((1,) + x.shape) 

                
                i = 0
                for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=outputDirname+folderName,
                              save_prefix=folderName + str(preNum), 
                              save_format='jpg'):
                    i += 1
                    if i > 10:
                        break  #mandatory
                print("Image number :{0} done!".format(j))
                j += 1
                preNum += 1
