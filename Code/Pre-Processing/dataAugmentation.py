import Augmentor
import shutil
import os
import random

#Function for data augmentation pipeline
#which takes an image path and augments the image within for n number of samples
def dataAugmentation(path,samples):
    p = Augmentor.Pipeline(path)
    p.rotate(probability=random.uniform(0.7,1), max_left_rotation=20, max_right_rotation=20)
    p.flip_left_right(probability=1)
    p.flip_top_bottom(probability= random.uniform(0.7,1))
    p.shear(probability=random.uniform(0.7,1),max_shear_right=20,max_shear_left=20)
    p.sample(samples)

#Function for renaming images based on consistent naming convention
def imageRename(filePath,label,db):
    folder = filePath
    for count, filename in enumerate(os.listdir(folder)):
        dst = f"AugIm{str(db)}_{str(count)}_{str(label)}.jpg"
        src = f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst = f"{folder}/{dst}"
        os.rename(src, dst)

#Function for main driver to run augmentation process
def AugmentationDriver(number_of_images,database):
    #List of folder names for IDB2
    fold_list = ["Norm Cells","Blast Cells"]

    # File path for images
    ALL_IDB2_Path = r"../../Dataset/ALL_IDB2/img"
    ALL_IDB1_Path = r"../../Dataset/ALL_IDB1/im/"
    ALL_IDB1_cropped_Path = r"../../Dataset/ALL_IDB1/cropped_aug/"

    #Augmentation for IDB1
    if(database =="IDB1"):
        for fold_name in fold_list:
            dataAugmentation(ALL_IDB1_Path + fold_name, number_of_images)
            imageRename(ALL_IDB1_Path + fold_name+"/output/", fold_list.index(fold_name), "IDB1")

    # Augmentation for IDB1 Cropped
    if (database == "IDB1_cropped"):
        for fold_name in fold_list:
            dataAugmentation(ALL_IDB1_cropped_Path + fold_name, number_of_images)
            imageRename(ALL_IDB1_cropped_Path + fold_name + "/output/", fold_list.index(fold_name), "IDB1")

    #Augmentation for IDB2
    elif(database == "IDB2"):
        for fold_name in fold_list:
            dataAugmentation(ALL_IDB2_Path + "/ALL_IDB2 AUG/" + fold_name, number_of_images)
            imageRename(ALL_IDB2_Path + "/ALL_IDB2 AUG/" + fold_name+"/output/", fold_list.index(fold_name), "IDB2")

AugmentationDriver(700,"IDB2")



