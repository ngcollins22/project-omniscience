import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os 
import time

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
    #start = time.time()
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
    if len(good)>6:
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
    
    #stop = time.time()
    #elapsed = stop-start
    #print("elapsed time: " + str(elapsed))
    return [[cx,cy], [refWidth, refHeight]]

def simulatedExperiment():
    root = os.getcwd()
    img2Path = os.path.join(root, "mars_4k_color.jpg") #training image

    # load images
    img2 = cv.imread(img2Path, cv.IMREAD_GRAYSCALE)

    #REPLACE WITH GROUND TRACK CORDS LATER
    refHeight, refWidth = img2.shape

    lon = np.array([12.953019303288272, 16.074598439627454, 19.886279227577894, 25.14641715819138, 32.66120938544461, 42.98064727347229, 55.64868716852323, 68.72627198552846, 79.95562364262398, 88.37620856286688, 94.30154842244991, 98.50805239009397, 101.78096268082827, 104.82747761829593, 108.32880740455279, 113.01575767098646, 119.6928123551966, 129.05791217032754, 141.08912679796524, 154.30776871834547, 166.34220974638146, 175.71153864122613, -172.91860854926836, -169.4160902737511, -166.36940981131315, -163.09733706143405, -158.8927652003549, -152.97056008300348, -144.55408743153387, -133.32856607396005, -120.25221979921726, -107.58183625569244, -97.25824617066145, -89.73959384094677, -84.47672388447145, -80.66348970907254, -77.54141168200931, -74.42032995383057, -70.61020090892474, -65.35279320985812, -57.84185974391276, -47.5265749004606, -34.86088310263406, -21.782065011932282, -10.548885087017851, -2.124187042069613, 3.8042893204901986, 8.012727105810232]) #
    lat = np.array([0.0, 10.294272932758956, 20.229582627857596, 29.367775107073633, 37.10643037065057, 42.628322369659436, 45.03715315047806, 43.817678288053095, 39.2420082806592, 32.142771222150316, 23.394353056120405, 13.670060408742872, 3.453488764018771, -6.879124038757107, -16.977353539563314, -26.438767988466793, -34.72492550380004, -41.08896303272316, -44.625362214642045, -44.628069893989036, -41.09647323585738, -34.735919972591915, -16.99186388850632, -6.894253544906227, 3.4382748595245154, 13.655278059914894, 23.380614021078312, 32.130909150700944, 39.233179799059656, 43.81324547712269, 45.03806161395708, 42.63436019720406, 37.11641518868723, 29.380372383101253, 20.24374726718599, 10.30925940310373, 0.015241728321373573, -10.279285698209046, -20.215416290406793, -29.35517483969081, -37.09644077122126, -42.622277725506194, -45.03623665686413, -43.82210371410697, -39.25083129393322, -32.15462976072552, -23.408090010845466, -13.684841709460322]) #
    degreetopix = refWidth/360
    lon = (lon + 180)*degreetopix
    lat = -(lat)*degreetopix + refHeight/2
    p = lon.argsort() #sort lon so it graphs right
    x=lon[p].astype(int)
    y=lat[p].astype(int)
    # x = np.arange(refWidth/4,refWidth*3/4, 50)
    
    # y = np.round(refHeight/4 * np.sin(x*np.pi/180 *0.2)+refHeight/2)
    
    L = np.round(refWidth/20)
    centerHistory=[]
    elapsedTimeHistory=[]
    for xi, yi in zip(x, y):
        
        # syntax is img[startY:endY, startX:endX]
        Yhigh = int(yi-L)
        Ylow = int(yi+L)
        Xleft = int(xi-L)
        Xright = int(xi+L)
        if Yhigh<0: Yhigh=int(0)
        if Ylow>refHeight: Ylow = int(refHeight)
        if Xleft<0:
            Xright = int(Xright-np.abs(Xleft))
            Xleft = 0
        if Xright>refWidth:
            Xleft = Xleft + (Xright-refWidth)
            Xright = refWidth

        croppedImage = img2[Yhigh:Ylow, Xleft:Xright]
        start = time.time()
        (center, refRes) = getPosePixelCords(croppedImage, numfeatures=50000)
        stop = time.time()
        centerHistory.append(center)
        elapsedTime = stop-start
        elapsedTimeHistory.append(elapsedTime)

        print(len(centerHistory))

    avgTimePerORB = np.mean(elapsedTimeHistory)
    print("average elapsed time per ORB call: " + str(avgTimePerORB))
    plt.figure()
    img2 = cv.imread(img2Path, cv.IMREAD_COLOR)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    plt.imshow(img2)

    plt.plot(x,y, color='green')

    centers = np.array(centerHistory, dtype=float) 
    cx = centers[:, 0]
    cy = centers[:, 1]
    plt.plot(cx,cy, 'r.-',  markersize=4)
    plt.legend(["True", "Estimated"])
    plt.xlabel("Horizontal Pixel Index (Proportional to Longitude)")
    plt.ylabel("Vertical Pixel Index (Proportional to Latitude)")
    plt.title("Simulated ORB Experiment")

    toggleRectangles = True
    if toggleRectangles:
        ax = plt.gca()
        for i in range(0, len(cx)):
            rect = patches.Rectangle((x[i]-L/2, y[i]-L/2), L, L, linewidth=1, edgecolor='b', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

    plt.show()

    #calculate error
    err = np.mean(np.sqrt((x-cx)**2 + (y-cy)**2))
    print(err)





    

if __name__ == '__main__':
    #ORB()
    #print(getPosePixelCords("mars_cropped_4k_1.jpg"))
    simulatedExperiment()