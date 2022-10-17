import keras.models
import numpy as np
import glob
import cv2
import os
from numpy import save
from sklearn.utils import shuffle
from keras.applications.densenet import DenseNet121
from keras.applications.efficientnet import EfficientNetB0
from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception

def exportModels(ModelName,SIZE):
       #Export pretrained networks
       CNN_model = EfficientNetB0(weights="imagenet", include_top = False, input_shape = (SIZE,SIZE,3))
       #We dont need to train parameters, we taking pre trained weights.
       for layer in CNN_model.layers:
           layer.trainable = False
       CNN_model.summary()
       CNN_model.save(r"./Models/"+ModelName+"_Model.h5")

#exportModels("EfficientNetB0",256)
def featureExtractor(SIZE,Database,imagePath):
    #FORMATTING FOR TRAINING
    dataset_images = []
    dataset_labels = []
    imgFormats = ("*.tif","*.jpg","*.jpeg")
    #All folder name within training directory will be the respective class
    for directory_path in glob.glob(imagePath+"/*"):
        label = directory_path.split("\\")[-1]
        print(label)
        #loop through all the images in the directory, finding images with .jpg and .tif
        for files in imgFormats:
            for img_path in glob.glob(os.path.join(directory_path,files)):
                print(img_path)
                #read images using cv2
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                #resize to 256x256 for VGG16
                img = cv2.resize(img, (SIZE,SIZE))
                #Convert colour channel
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                dataset_images.append(img)
                dataset_labels.append(label)

    #Converting to numpy arrays
    dataset_images = np.array(dataset_images)
    dataset_labels = np.array(dataset_labels)
    dataset_images_shuffled, dataset_labels_shuffled = shuffle(dataset_images, dataset_labels)

    # Saving labels into a file
    save(r"./"+Database+"/"+Database+"_labels.npy", dataset_labels_shuffled)
    print(dataset_labels_shuffled)
    #Normalizing
    dataset_images_shuffled = dataset_images_shuffled/255.0

    #Loading models
    models = {}
    VGG16_loaded = keras.models.load_model(r"./Models/VGG16_Model.h5", compile=False)
    models["VGG16"] = VGG16_loaded
    DenseNet121_loaded = keras.models.load_model(r"./Models/DenseNet121_Model.h5", compile=False)
    models["DenseNet121"] = DenseNet121_loaded
    xCeption_loaded = keras.models.load_model(r"./Models/xCeption_Model.h5", compile=False)
    models["xCeption"] = xCeption_loaded
    EfficientNetB0_loaded = keras.models.load_model(r"./Models/EfficientNetB0_Model.h5", compile=False)
    models["EfficientNetB0"] = EfficientNetB0_loaded

    #Loop through each pretrained CNN and extract features, flatten, then save the file
    for model_name, model in models.items():
        print(model.summary())
        feature_extractor = model.predict(dataset_images_shuffled)
        print(feature_extractor.shape)
        X_features = feature_extractor.reshape(feature_extractor.shape[0],-1)
        save(os.path.join(r"./"+Database+"/Features/", model_name + "_"+Database+"_features.npy"), X_features)

featureExtractor(256, "IDB1_cropped", "../../Dataset/ALL_IDB1/cropped_aug")
