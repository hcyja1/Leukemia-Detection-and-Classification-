import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
from tkinter import ttk
from tkinter import filedialog
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import time as t
import threading
from numpy import load
from keras.models import load_model
from sklearn import preprocessing
import imutils
import cv2
import pickle

################ DATASET, VIRTUAL ENVIRONMENT, FEATURE DOWNLOAD ########################
########## PLEASE REFER TO README.TXT FILE FOR CORRECT DIRECTORY SET UP ############

#Label Encodder and decoder
def loadLabelEncoder():
    labels = load(r"../Code/Feature Extraction/"+database + "/" + database + "_labels.npy")
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    return le

#Load pretrained CNNs
def loadCNN(modelName):
    loaded_model = load_model(r"../Code/Feature Extraction/Models/" + modelName + "_Model.h5")
    return loaded_model

#Load XGB Classifier
def loadClassifier():
    classification_model = pickle.load(open("../Code/Models/"+database+"/Ensemble Model_XGB_" + database+"_Model.pkl", "rb"))
    return classification_model

#Open Coordinate file corresponding to image
def openXYC(file_name):
    coordinate_arr = []
    xyc_path = r"../Dataset/ALL_IDB1/xyc/" + file_name
    with open(xyc_path,"r") as coordinate_file:
        for coordinate in coordinate_file:
            formatted_coordinate = coordinate.split()
            coordinate_arr.append(formatted_coordinate)
    return coordinate_arr

#Convert and display output image
def convertImage(image,img_size):
    image_convert= Image.fromarray(image)
    wpercent = (img_size / float(image_convert.size[0]))
    hsize = int((float(image_convert.size[1]) * float(wpercent)))
    convert_labels = image_convert.resize((img_size, hsize), Image.ANTIALIAS)
    imgtk_2 = ImageTk.PhotoImage(image=convert_labels)
    return imgtk_2

#Function to load image to GUI
def load_img(database_selected):
    #Reset progress and define globals
    progress['value'] = 0
    global img, image_data,database,ground_truth,basewidth
    database = database_selected
    for img_display in frame.winfo_children():
        img_display.destroy()
    #Set default open paths depending on which button is clicked
    if(database_selected == "IDB1"):
        initial_dir = r"../Dataset/ALL_IDB1/demo"
        basewidth = 350
    elif(database_selected=="IDB1_cropped"):
        initial_dir=r"../Dataset/ALL_IDB1/im"
        basewidth=250
    elif(database_selected=="IDB2"):
        initial_dir="../Dataset/ALL_IDB2/img/Demo"
        basewidth = 350

    image_data = filedialog.askopenfilename(initialdir=initial_dir, title="Choose an image",
                                       filetypes=(("all files", "*.*"), ("png files", "*.png")))
    ground_truth = image_data.split("/")[-1].split("_")[-1].split(".")[0]

    #Resize images
    img = Image.open(image_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')

    #If not object detection,do this
    if (database_selected != "IDB1_cropped"):
        panel = tk.Label(frame, pady=15, bg="white", text=str(file_name[len(file_name) - 1]).upper()).pack()
        panel_image = tk.Label(frame, image=img).pack()
    #Else do this
    else:
        panel = tk.Label(frame,bg="white", text=str(file_name[len(file_name) - 1]).upper()).grid(row=1, column=0)
        panel_image = tk.Label(frame, image=img).grid(row=0, column=0,pady=(10,0),padx=(50,15))

#Set button states to disabled or enabled
def buttonStates(state):
    detect_btn["state"] = state
    choose_image_IDB1["state"] = state
    choose_image_IDB2["state"] = state
    choose_image_IDB1_cropped["state"] = state

#Function to make sure progress does not exceed 90% if function not complete
def progressCheck(progress_bool):
    if (progress_bool == False and progress['value'] >= 90):
        progress['value'] == 90

def classify():
    #Define globals
    global displayThresh, displayResults, bChannel, displaySegmented
    #Disable button, make progress to 0, set GUI displays
    buttonStates("disable")
    progress['value'] = 0
    caption_height=0
    image_height=10
    image_gap = 15
    # loading ensemble model
    X1 = loadClassifier()
    # load Label Encoder
    le = loadLabelEncoder()


    if(database=="IDB1_cropped"):
        #Start counter
        tic = t.perf_counter()
        #Define boolean for progressbar control
        process_complete = False

        # read images using cv2
        image = cv2.imread(image_data, cv2.IMREAD_COLOR)
        resultsImage = image.copy()
        roundSegmentationImage = image.copy()
        num_blast_cell=0
        coordinate_file = ((image_data.split("/")[-1]).split(".", 1))[0] + ".xyc"
        coordinates = openXYC(coordinate_file)
        if (coordinate_file.split("_")[-1].split('.')[0] == '1'):
            for coordinate in coordinates:
                blast_x = coordinate[0]
                blast_y = coordinate[1]
                cv2.putText(resultsImage, "X", (int(blast_x), int(blast_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        progress['value'] += 3
        # convert the mean shift image
        shifted = cv2.pyrMeanShiftFiltering(image, 31, 51)
        progress['value'] += 5
        # Convert to L*A*B, Extract B
        lab = cv2.cvtColor(shifted, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        progress['value'] += 5
        #Display b Channel on GUI
        bChannel = convertImage(b,basewidth)
        b_tk = tk.Label(frame, image=bChannel).grid(row=0,column=1,padx=image_gap,pady=(image_height,caption_height))
        panel = tk.Label(frame, bg="white", text="B Channel Extraction").grid(row=1,column=1)
        # Otsu Thresholding
        thresh = cv2.threshold(b, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        progress['value'] += 5
        # Display thresholding on GUI
        displayThresh = convertImage(thresh, basewidth)
        watershed_tk = tk.Label(frame, image=displayThresh).grid(row=0, column=2, padx=image_gap,pady=(image_height,caption_height))
        panel = tk.Label(frame, bg="white", text="Otsu Thresholding").grid(row=1, column=2)
        progress['value'] += 5
        # compute the exact Euclidean distance from every binary
        # pixel to the nearest zero pixel, then find peaks in this
        # distance map
        D = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(D, indices=False, min_distance=25,
                                  labels=thresh)
        progress['value'] += 5
        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then apply the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)

        print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
        progress['value'] += 5
        # loop over the unique labels returned by the Watershed
        # algorithm
        for label in np.unique(labels):
            # if the label is zero, we are examining the 'background'
            # so simply ignore it
            if label == 0:
                continue
            # otherwise, allocate memory for the label region and draw
            # it on the mask
            mask = np.zeros(b.shape, dtype="uint8")
            mask[labels == label] = 255
            # detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)

            #Define bounding rectangles
            xr, yr, wr, hr = cv2.boundingRect(c)
            center_x, center_y = xr + (wr / 2), yr + (hr / 2)

            # draw a circle enclosing the object
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(roundSegmentationImage, (int(x), int(y)), int(r), (0, 255, 0), 2)

            # Filter out small objects
            (H, W) = image.shape[:2]
            if (wr) / float(W) < 0.02 or (hr) / float(H) < 0.02:
                continue

            # crop out
            crop = image[yr:yr + hr, xr:xr + wr]
            crop = cv2.resize(crop, (256, 256))

            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            crop = np.expand_dims(crop, axis=0)
            crop = crop / 255

            #Predict features
            f1 = C1.predict(crop)
            f1_features = f1.reshape(f1.shape[0], -1)
            f2 = C2.predict(crop)
            f2_features = f2.reshape(f2.shape[0], -1)
            ensemble_features = np.concatenate([f1_features, f2_features], 1)
            prediction = X1.predict(ensemble_features)
            decoded_result = le.inverse_transform(prediction)

            if (decoded_result == "Blast Cells"):
                decoded_result = "ALL"
                num_blast_cell = num_blast_cell + 1
                print(center_x, center_y)
                cv2.rectangle(resultsImage, (xr, yr), (xr + wr, yr + hr), (0, 255, 0), 2)
            else:
                decoded_result = "Norm"
                cv2.putText(resultsImage, "{}".format(decoded_result), (int(center_x), int(center_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(resultsImage, (xr, yr), (xr + wr, yr + hr), (0, 0, 255), 2)

            # Display segmented on GUI
            roundSegmentationImage_RGB = cv2.cvtColor(roundSegmentationImage, cv2.COLOR_BGR2RGB)
            displaySegmented = convertImage(roundSegmentationImage_RGB, basewidth)
            segmented_tk = tk.Label(frame, image=displaySegmented).grid(row=0, column=3, padx=image_gap,
                                                                        pady=(image_height, caption_height))
            panel = tk.Label(frame, bg="white", text="Segmented").grid(row=1, column=3)

            resultsImage_RGB = cv2.cvtColor(resultsImage, cv2.COLOR_BGR2RGB)
            displayResults = convertImage(resultsImage_RGB, 520)
            finalresults_tk = tk.Label(frame, image=displayResults).grid(row=2, column=0, columnspan=3,
                                                                         pady=(15, 0))
            panel = tk.Label(frame, bg="white", text="Final Results").grid(row=3, column=0, columnspan=3,
                                                                           pady=(5, 0))
            progress['value'] += 0.7
            progressCheck(process_complete)

        if(num_blast_cell>0):
            leukemia_detected = "True"
        else:
            leukemia_detected = "False"
        toc = t.perf_counter()
        panel = tk.Label(frame, bg="#1a296b",fg="white", font=("", 13),text="Time Taken : {:.2f} seconds \n\n Leukemia Detected : {} \n\nPredicted : {} Blast Cells"
                         .format(toc-tic,leukemia_detected, num_blast_cell)).grid(padx=15,row=2, column=2, columnspan=3)
        process_complete = True
        progress['value'] = 100
    else:
        tic = t.perf_counter()
        # Load Image

        img = cv2.imread(image_data, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        input_img = np.expand_dims(img, axis=0)
        input_img = input_img / 255

        # Extract features from image
        f1 = C1.predict(input_img)
        f1 = f1.reshape(f1.shape[0], -1)
        f2 = C2.predict(input_img)
        f2 = f2.reshape(f2.shape[0], -1)
        ensemble_features = np.concatenate([f1, f2], 1)
        progress['value'] += 50
        # make prediction
        prediction = X1.predict(ensemble_features)

        #Check if prediction is correct, if yes, make green background, else put red
        if (str(prediction[0])!=ground_truth):
            background="green"
        else:
            background="red"
        prediction = le.inverse_transform(prediction)

        #Format display string
        if (prediction[0] == "Blast Cells"):
           prediction_format = "Leukemia Patient"
        else:
            prediction_format = "Non Leukemia Patient"
        #End clock
        toc = t.perf_counter()

        #Display results on GUI
        result = tk.Label(frame,text= "Prediction : {} Detected".format(str(prediction_format)),pady=5,font=("",12),bg=background).pack()
        time = tk.Label(frame, text="Time taken : {:.2f}".format(toc-tic), pady=5, font=("", 12),bg = "white").pack()

    #Make button state normal and progress bar goes to 100%, indicating complete
    progress['value'] = 100
    buttonStates("normal")
    return
#Start GUI
def GUI():
    global frame,C1,C2,detect_btn,choose_image_IDB1,choose_image_IDB1_cropped,choose_image_IDB2,progress

    # Loading CNNs
    C1 = loadCNN("VGG16")
    C2 = loadCNN("DenseNet121")

    #Main Driver
    root = tk.Tk()
    root.resizable(False, False)

    #Define Title
    root.title('Acute Lymphoblastic Leukemia Detector')
    title = tk.Label(root, text="Acute Lymphoblastic Leukemia Detector", padx=25, pady=15, font=("", 12)).pack()

    #Define Canvas
    canvas = tk.Canvas(root, height=700, width=1300)
    canvas.pack()

    #Define Frame
    frame = tk.Frame(root, bg='white')
    frame.place(relwidth=0.90, relheight=0.87, relx=0.05, rely=0.07)

    ##Add Button for Object Detection
    objd_title = tk.Label(root,text= "Detection : ",pady=10,padx=0,font=("",12))
    objd_title.pack(side=tk.LEFT,padx=(50,0))
    #IDB1_cropped
    choose_image_IDB1_cropped = tk.Button(root, text='Full Blood Smear Image',
                            padx=25, pady=10,
                            fg="white", bg="grey", command=lambda:load_img("IDB1_cropped"))
    choose_image_IDB1_cropped.pack(side=tk.LEFT)

    ##Add Buttons for Classification
    classf_title = tk.Label(root,text= "Classification : ",pady=10,padx=8,font=("",12))
    classf_title.pack(side=tk.LEFT)
    #IDB1
    choose_image_IDB1 = tk.Button(root, text='Full Blood Smear Image',
                            padx=25, pady=10,
                            fg="white", bg="grey", command=lambda:load_img("IDB1"))
    choose_image_IDB1.pack(side=tk.LEFT)
    #IDB2
    choose_image_IDB2 = tk.Button(root, text='Individual Cell',
                            padx=25, pady=10,
                            fg="white", bg="grey", command=lambda:load_img("IDB2"))
    choose_image_IDB2.pack(side=tk.LEFT)

    #Classification Button
    detect_btn = tk.Button(root, text='Detect',
                            padx=35, pady=10,
                            fg="white", bg="grey", command=lambda:threading.Thread(target=classify).start())
    detect_btn.pack(side=tk.RIGHT)

    #Progress Bar
    progress_title = tk.Label(root, text="Progress : ", pady=10, padx=8, font=("", 12)).pack(side=tk.LEFT)
    progress = ttk.Progressbar(root,orient=tk.HORIZONTAL,length=270,mode='determinate')
    progress.pack(side=tk.LEFT)

    root.mainloop()

GUI()