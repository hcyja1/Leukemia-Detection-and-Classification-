# Leukemia-Detection-and-Classification-
Detection and Classification of Acute Lymphoblatic Leukemia using Hybrid Ensemble CNN XGBoost model optimized by Genetic Algorithm


Full Report with Methodology and Experimental Details : https://drive.google.com/file/d/1BrS_4uu9_MWsD8sakjv855hT7L_s9dHX/view?usp=sharing


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

Dropbox links : 

1) Features(For testing model accuracy) : https://uniofnottm-my.sharepoint.com/:u:/g/personal/hcyja1_nottingham_ac_uk/ETfUgMrKfD1GsejE8Yt-KO0Bk7wtYlJJL9ZEdJ_tVQtR_w?e=nVV0r8
2) Virtual Environment & Dataset : https://uniofnottm-my.sharepoint.com/:u:/g/personal/hcyja1_nottingham_ac_uk/EamF0G_FiBpPsdkqWAvkXG8BAhX942c0vP8tVZMDKwgPzg?e=tABHpk
3) Folder with both : https://uniofnottm-my.sharepoint.com/:f:/g/personal/hcyja1_nottingham_ac_uk/EnH4pSzuNzpBq_NfdfsIA4UBAvtt_f8ewYEUa31ZexWyPA?e=ozqk5i

Instructions on running ALL-CAD Diagnostic System : 
1) Open command prompt(cmd) 
2) Use "cd" to change to directory of "../Implementation/environment-env/Scripts/
3) Activate virtual environment by typing in the command of "activate.bat" into your cmd.
4) Use "cd" once again to change to the directory of "../Implementation/ALL-CD-System/".
5) Run the command "python ALL-CAD-System.py".

Instructions on use : 
1) The software functionality is mainly split into 3 sections,
   a. ALL Detection
   b. Full blood smear image classification
   c. Individual lymphocyte classification
2) On clicking a/b/c in the root of the GUI interface, a pop up will indicating the selection of an image will show up. The default directory is set to the corresponding image type. 
3) After selection of image, press the "detect" button on the far right.
4) Results will be shown on scren. 

Generally, please take note of these things:
1) Activate the virtual environment created before running the ALL-CAD-System.py file. This allows packages and dependencies to be handled. 
2) Ensure the dataset directory is properly handled.

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

