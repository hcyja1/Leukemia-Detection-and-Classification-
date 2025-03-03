# Leukemia-Detection-and-Classification-
Detection and Classification of Acute Lymphoblatic Leukemia using Hybrid Ensemble CNN XGBoost model optimized by Genetic Algorithm

(Full Report with Methodology and Experimental Details)[https://drive.google.com/file/d/1GZy2mB0nXI1fb9Xol928lVeD4KaCM0hc/view?usp=sharing]

Supplementary item details and download : 
IMPORTANT : Please put environment-env and Dataset in the same directory as the ALL-CAD-System and Code.
The directory should be : 
20150707_software(main folder)
	-> ALL-CAD-System
	-> Code
		-> Feature Extraction(From dropbox link 1, please extract here to test accuracy)
	-> Dataset(From dropbox link 2, please extract dataset in main folder)
		-> ALL IDB1
		-> ALL IDB2
	-> environment-env(From dropbox link 2, please extract dataset in main folder)


"Code" File Directories and usage : 
1) Feature Extraction - Contains pretrained CNN models used for feature extraction, extracted features and corresponding labels, feature extraction code. 
2) Models - Contains models used for object detection and classification of individual and full blood smear images, and also the code for training XGB models.
3) Pre-Processing - Contains code for augmentation and code to build IDB1_cropped dataset.
4) Tuning - Contains code for GA hyperparameter optimization.


Main Libraries used:
1) Keras
2) sklearn
3) Tensorflow
4) numpy
5) opencv-python
6) xgboost
7) seaborn
8) tkinter
9) pickle 
10) pillow
11) Augmentor

![Screenshot (134)](https://user-images.githubusercontent.com/73547478/209740314-37c969dc-30bb-441e-a361-bf77a27496a6.png)
![Screenshot (135)](https://user-images.githubusercontent.com/73547478/209740320-87ee80c0-15eb-4fe9-b110-bf1c54accdbc.png)
![Screenshot (138)](https://user-images.githubusercontent.com/73547478/209740323-59cef5d1-9644-4461-99b4-6fdecdc8a643.png)
![Screenshot (137)](https://user-images.githubusercontent.com/73547478/209740329-d3b32489-2852-483e-b810-849fa0f6fa94.png)



Notable documentations:
1) https://sklearn-genetic-opt.readthedocs.io/en/stable/tutorials/basic_usage.html
2) https://xgboost.readthedocs.io/en/stable/python/python_api.html
3) https://augmentor.readthedocs.io/en/master/

