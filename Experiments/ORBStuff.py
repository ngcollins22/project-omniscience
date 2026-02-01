import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os 

# Code largely sourced from https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html,  https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html, 
# and https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html

# i had to make an environment to run it. need to have it all be in the omni environment at some point
# conda create -n marsorb python=3.11 numpy<2 matplotlib -y
# conda activate marsorb

def ORB():
    root = os.getcwd()
    img1Path = os.path.join(root, "mars_cropped_4k_1.jpg") #query image
    img2Path = os.path.join(root, "mars_4k_color.jpg") #training image
    #download image here https://planetpixelemporium.com/mars5672.html#
    #crop it for the query
    # Seems to work best when the resolution of both images are the same, so we need to use 4k i think to match arducam

    # load images
    img1 = cv.imread(img1Path, cv.IMREAD_GRAYSCALE) #query: This is what we are trying to match to the map
    img2 = cv.imread(img2Path, cv.IMREAD_GRAYSCALE) #Reference or Training: This is the perfect entire map

    #initiate orb
    orb = cv.ORB_create(nfeatures=50000) #increasing number of keypoints from default

    # get keypoints and descriptors for each image
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher (Brute Force Matcher) object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False) 

    # get matches using knn matching. finds the k best matches per keypoint and the filter below only keeps the 
    # match if the 1st best match is significantly better than the 2nd best match
    # Was previously  using:  matches = bf.match(des1,des2) and no filter
    knn = bf.knnMatch(des1, des2, k=2) 
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good.append(m)
 
    # Sort them in the order of their distance. This isn't really necessary anymore with the filter but still good practice.
    good = sorted(good, key = lambda x:x.distance)

    # Using homography to outline the query image
    if len(good)>10:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
 
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
 
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        #get center of rectangle which is where the camera would be under ideal conditions
        pts = dst.reshape(-1, 2)
        center = pts.mean(axis=0)
        cx, cy = center
        print("Center (x, y):", cx, cy)

        #plot center and query image outline
        img2 = cv.circle(img2, (cx.astype(int), cy.astype(int)), radius=6, color=(255,0,255), thickness=-1)
        img2 = cv.polylines(img2,[np.int32(dst)],True, 255, 3, cv.LINE_AA)
 
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), 10) )
        matchesMask = None

    # Draw matches
    img3 = cv.drawMatches(img1,kp1,img2,kp2, good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.title("ORB Brute Force Feature Matching With " + str(len(good)) + " Keypoints", )
    plt.imshow(img3),plt.show()

    
    #visualize keypoints on train image. toggle on or off below
    toggleKpPlot = True
    if toggleKpPlot:
        kpPlotImg= cv.drawKeypoints(img2, kp2, img2, color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure()
        plt.imshow(kpPlotImg)
        plt.show() 


"""simple function that takes in a query image and returns the center of where it is in the hardcoded reference image
return: a tuple containing first a list of the pixel cords of the center of the query image, and second the resolution
 of the reference image so another function can use that info to convert the scale"""
def getPosePixelCords(img1,numfeatures=50000) -> tuple[list, list]:
    root = os.getcwd()
    #img1Path = os.path.join(root, queryImageFilePath) #query image
    img2Path = os.path.join(root, "mars_4k_color.jpg") #training image
    #download image here https://planetpixelemporium.com/mars5672.html#

    # load images
    #img1 = cv.imread(img1Path, cv.IMREAD_GRAYSCALE) #query: This is what we are trying to match to the map
    img2 = cv.imread(img2Path, cv.IMREAD_GRAYSCALE) #Reference or Training: This is the perfect entire map
    refHeight, refWidth = img2.shape

    #initiate orb
    orb = cv.ORB_create(nfeatures=numfeatures) #increasing number of keypoints from default

    # get keypoints and descriptors for each image
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher (Brute Force Matcher) object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False) 

    # get matches using knn matching. finds the k best matches per keypoint and the filter below only keeps the 
    # match if the 1st best match is significantly better than the 2nd best match
    # Was previously  using:  matches = bf.match(des1,des2) and no filter
    knn = bf.knnMatch(des1, des2, k=2) 
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # Using homography
    if len(good)>10:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
 
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
 
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        #get center of rectangle which is where the camera would be under ideal conditions
        pts = dst.reshape(-1, 2)
        center = pts.mean(axis=0)
        cx, cy = center
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), 10) )
        matchesMask = None
        return   [[np.nan,np.nan], [refWidth, refHeight]]
    

    return [[cx,cy], [refWidth, refHeight]]

def syntheticExperiment():
    root = os.getcwd()
    img2Path = os.path.join(root, "mars_4k_color.jpg") #training image

    # load images
    img2 = cv.imread(img2Path, cv.IMREAD_GRAYSCALE)

    #REPLACE WITH GROUND TRACK CORDS LATER
    refHeight, refWidth = img2.shape
    x=np.arange(1,refWidth, 10)
    y = np.round(refHeight/4 * np.sin(x*np.pi/180)+refHeight/2)

    L = 300
    centerHistory=[]
    for xi, yi in zip(x, y):
        
        # syntax is img[startY:endY, startX:endX]
        Yhigh = int(yi-L)
        Ylow = int(yi+L)
        Xleft = int(xi-L)
        Xright = int(xi+L)
        if Yhigh<0: Yhigh=int(0)
        if Ylow>refHeight: Ylow = int(refHeight-1)
        if Xleft<0: Xleft = int(0)
        if Xright>refWidth: Xright = int(refWidth-1)

        croppedImage = img2[Yhigh:Ylow, Xleft:Xright]
        (center, refRes) = getPosePixelCords(croppedImage, numfeatures=50000)
        centerHistory.append(center)
        print(len(centerHistory))

    plt.figure()
    plt.imshow(img2)
    plt.gca().invert_yaxis()
    plt.plot(x,y)

    centers = np.array(centerHistory, dtype=float) 
    cx = centers[:, 0]
    cy = centers[:, 1]
    plt.plot(cx,cy)
    plt.show()






    

if __name__ == '__main__':
    #ORB()
    #print(getPosePixelCords("mars_cropped_4k_1.jpg"))
    syntheticExperiment()