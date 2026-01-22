import cv2 as cv
import matplotlib.pyplot as plt
import os 

# Code largely sourced from https://www.youtube.com/watch?v=0sPlnrEMyYk
# and https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html

# i had to make an environment to run it
# conda create -n marsorb python=3.11 numpy<2 matplotlib -y
# conda activate marsorb


def ORB():
    root = os.getcwd()
    imgPath = os.path.join(root, "mars_1k_color.jpg")
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    orb = cv.ORB_create()
    kp = orb.detect(img,None)
 
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    img = cv.drawKeypoints(img, kp, img, color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure()
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    ORB()