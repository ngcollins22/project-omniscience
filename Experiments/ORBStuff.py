import cv2 as cv
import matplotlib.pyplot as plt
import os 

# Code largely sourced from https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html and https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

# i had to make an environment to run it. need to have it all be in the omni environment at some point
# conda create -n marsorb python=3.11 numpy<2 matplotlib -y
# conda activate marsorb

def ORB():
    root = os.getcwd()
    trainImgPath = os.path.join(root, "../mars_1k_color.jpg") 
    queryImgPath = os.path.join(root, "../CroppedMars.png")


    # load images
    imgTrain = cv.imread(trainImgPath, cv.IMREAD_GRAYSCALE) #This is the perfect entire map
    imgQuery = cv.imread(queryImgPath, cv.IMREAD_GRAYSCALE) #This is what we are trying to match to the map

    #initiate org
    orb = cv.ORB_create()

    # get keypoints and descriptors for each image
    kpTrain, desTrain = orb.detectAndCompute(imgTrain, None)
    kpQuery, desQuery = orb.detectAndCompute(imgQuery,None)

    # create BFMatcher (Brute Force Matcher) object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True) #can add a KNN method to 
 
    # Match descriptors.
    matches = bf.match(desQuery,desTrain)
 
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
 
    # Draw first 10 matches.
    img3 = cv.drawMatches(imgQuery,kpQuery,imgTrain,kpTrain,matches[:15],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
    plt.imshow(img3),plt.show()

    # #visualize keypoints on train image
    # trainImg = cv.drawKeypoints(trainImg, kpTrain, trainImg, color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.figure()
    # plt.imshow(trainImg)
    # plt.show()

if __name__ == '__main__':
    ORB()