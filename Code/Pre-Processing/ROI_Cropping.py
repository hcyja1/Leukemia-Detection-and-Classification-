# import the necessary packages
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import glob
import imutils
import cv2

#Function to count total number of blast cells
def blastCellCounter():
    blast_cells = 0

    # Go through all xyc files
    for img_path in glob.glob("../../../Dataset/ALL_IDB1/xyc/*"):
        #If XYC is labelled "1", meaning there are blast cells,
        coordinate_file_label = img_path.split("_")[-1].split(".", 1)[0]
        if(coordinate_file_label=="1"):
            #Loop through the input, each line represents a blast cell
            with open(img_path, "r") as coordinate_file:
                for coordinate in coordinate_file:
                    #Count and return blsat cells
                    blast_cells = blast_cells+1
    return blast_cells

#Function for opening coordinate files
def openXYC(file_name):
    #Array to store coordinates
    coordinate_arr = []
    #Coordinate file path
    xyc_path = r"../../../Dataset/ALL_IDB1/xyc/" + file_name
    with open(xyc_path,"r") as coordinate_file:
        #Format each line in the file and append to coordinate array
        for coordinate in coordinate_file:
            formatted_coordinate = coordinate.split()
            coordinate_arr.append(formatted_coordinate)
    return coordinate_arr

#Function to add margin error and to check if segmented prospect is a blast cell corresponding to coordinate file
def rangeChecker(xyc_file,margin,center_x,center_y):
    coordinates = openXYC(xyc_file)

    for coordinate in coordinates:
        #X-Coordinate within file, range 1 and range 2 represents values within +-x range of x-coordinate
        blast_x = int(coordinate[0])
        blast_x_range1 = blast_x-margin
        blast_x_range2 = blast_x+margin
        x_range = range(blast_x_range1, blast_x_range2, 1)

        # Y-Coordinate within file, range 1 and range 2 represents values within +-x range of y-coordinate
        blast_y = int(coordinate[1])
        blast_y_range1 = blast_y - margin
        blast_y_range2 = blast_y + margin
        y_range = range(blast_y_range1,blast_y_range2,1)

        #check if extracted x and y coordinates are within +20 of range
        if(center_x in x_range and center_y in y_range):
            print("Checking {} : ({},{}) : Blast Cells!".format(coordinate,center_x,center_y))
            return True

#Function to create new dataset
def croppingROI():

    blast_cell_count=0
    norm_cell_count=0

    # All folder name within training directory will be the respective class
    for img_path in glob.glob("../../../Dataset/ALL_IDB1/im/*"):

        coordinate_file = ((img_path.split("\\")[-1]).split(".",1))[0] + ".xyc"
        coordinate_file_label = coordinate_file.split("_")[-1].split(".", 1)[0]

        # read images using cv2
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # convert the mean shift image
        shifted = cv2.pyrMeanShiftFiltering(image, 30, 60)

        # Extract b channel from LAB

        lab = cv2.cvtColor(shifted, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        thresh = cv2.threshold(b, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]



        # compute the exact Euclidean distance from every binary
        # pixel to the nearest zero pixel, then find peaks in this
        # distance map
        D = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(D, indices=False, min_distance=25,
                                  labels=thresh)
        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)
        print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

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

            #Calculate bounding rectangle
            xr, yr, wr, hr = cv2.boundingRect(c)
            center_x,center_y = xr+(wr/2), yr+(hr/2)

            #Filter out small objects
            (H, W) = image.shape[:2]
            if (wr) / float(W) < 0.02 or (hr) / float(H) < 0.02:
                continue

            #crop out
            crop = image[yr:yr + hr, xr:xr + wr]
            crop = cv2.resize(crop, (256, 256))

            #Blast cells are present in image
            if(coordinate_file_label=="1"):
                #Check coordinates
                coordinateCheck = rangeChecker(coordinate_file, 42, round(center_x), round(center_y))
                #Blast Cell
                if(coordinateCheck is True):
                    blast_cell_count = blast_cell_count+1
                    cv2.imwrite("../../../Dataset/ALL_IDB1/cropped/Blast Cells/Im_" + str(blast_cell_count)+"_1.jpg", crop)
                #Not a blast cell
                else:
                    norm_cell_count = norm_cell_count+1
                    cv2.imwrite("../../../Dataset/ALL_IDB1/cropped/Norm Cells/" + str(norm_cell_count)+"_0.jpg", crop)

            #No Blast cells present in entire image
            elif(coordinate_file_label=="0"):
                norm_cell_count = norm_cell_count+1
                cv2.imwrite("../../../Dataset/ALL_IDB1/cropped/Norm Cells/" + str(norm_cell_count) + "_0.jpg", crop)


    BlastCellCounter = blastCellCounter()
    print("Found {} Blast Cells. Actual Blast Cells : {}".format(blast_cell_count, BlastCellCounter))

croppingROI()
