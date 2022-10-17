import cv2
from numpy import load
from sklearn import preprocessing
import numpy as np
from keras.models import load_model
import pickle
import glob
import os

def loadModel(modelName):
    loaded_model = load_model(r"../../Feature Extraction/Models/" + modelName + "_Model.h5")
    return loaded_model

def predictImage():
    labels = load(r"../../Feature Extraction/IDB2/IDB2_labels.npy")
    le = preprocessing.LabelEncoder()
    le.fit(labels)

    #Load Models
    CNN1 = loadModel("DenseNet121")
    CNN2 = loadModel("VGG16")

    #Load classification
    ensembleModel = pickle.load(open("Ensemble Model_XGB_IDB2_Model.pkl","rb"))
    
    imgFormats = ("*.tif", "*.jpg", "*.jpeg")
    Size = 256

    for directory_path in glob.glob(r"../../../Dataset/ALL_IDB2/img/Demo/*"):
      label = directory_path.split("\\")[-1]
      print("--------------------------" + label + "----------------------")
      for files in imgFormats:
        for img_path in glob.glob(os.path.join(directory_path,files)):

            # Load Image
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (Size, Size))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            input_img = np.expand_dims(img, axis=0)
            input_img = input_img/255

            # Extract features from image
            f1= CNN1.predict(input_img)
            f1 = f1.reshape(f1.shape[0], -1)

            f2 = CNN2.predict(input_img)
            f2 = f2.reshape(f2.shape[0], -1)

            ensemble_features = np.concatenate([f1,f2], 1)

            # make prediction
            prediction = ensembleModel.predict(ensemble_features)
            print("Predicted - {}".format(le.inverse_transform(prediction)))

predictImage()


