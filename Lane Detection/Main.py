from P1 import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Initializing the parameters
kernel = 5                      # Kernel size for canny Edge Detection
lowThreshold = 50               # Low Threshold value for Canny
highThreshold = 150             # High Threshold value for Canny
maskRegion = np.array([[(87,540),(462, 313), (512, 315), (954,540)]], dtype=np.int32) # Attempted through hit and trial method
rho = 1                         # Distance resolution in pixels of the Hough grid
theta = np.pi/180               # Angular Resolution in radians of the Hough grid
houghThreshold = 20             # Minimum number of votes, intersection in Hough grid cell
line_length = 20                # Minimum no of pixels making up the line
max_linegap = 320               # Maximum pixel gap in pixels between the connectable line segments
alpha = 0.8
beta = 1.
gamma = 0.

def drawLinesPipeline(true_image, kernel, lowThreshold, highThreshold, rho, theta, houghThreshold, line_length, max_linegap):
    img_copy = np.copy(true_image)                  # making a duplicate copy of the image to prevent any unintended modifications 
    gray_img = convert2Gray(img_copy)               # Converting the image to grayscale
    ImgBlur = applyBlur(gray_img, kernel)           # Applying blur on the image to smoothen the hard edges

    cannyImg = applyCanny(ImgBlur, lowThreshold, highThreshold)         # Applying Canny Edge Detection on the smoothened image
    imgROI = RegionOfInterest(cannyImg, maskRegion)                     # Fetching the region of interest to reduce computation 
    houghImg = applyHough(imgROI, rho, theta, houghThreshold, line_length, max_linegap)         # Applying the Hough transform
    weightedImg = fetchWeightedImage(img_copy, houghImg, alpha, beta, gamma)                    # Final weighted Image

    finalImg = weightedImg

    # convertedImg = cv2.cvtColor(finalImg, cv2.COLOR_BGR2RGB)  
    # plt.imshow(convertedImg)
    
    # When read from cv2, it gives an BGR format output, which is why
    # we had to convert to RGB for plt.imshow() which accepts RGB image.
    
    plt.imshow(finalImg)
    plt.show()

if __name__ == "__main__":

    true_image = mpimg.imread("../data/SDC_Term1_P1_FindingLanes/test_images/solidYellowCurve.jpg")
    result =  drawLinesPipeline(true_image, kernel, lowThreshold, highThreshold, rho, theta, houghThreshold, line_length, max_linegap)