import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

# Applying Grayscale
# Applying Gaussian Blur
# Applying Canny
# Applying Region_Of_Interest
# Applying DrawLines
# Fetching Weighted Image

# Applying Grayscale
def convert2Gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray

# Applying Gaussian Blur
def applyBlur(img, kernel):
    blurImg = cv2.GaussianBlur(img, (kernel, kernel), 0)
    return blurImg


# Applying Canny
def applyCanny(img, lowThreshold, highThreshold):
    cannyImg = cv2.Canny(img, lowThreshold, highThreshold)
    return cannyImg

# Applying Region_Of_Interest
def RegionOfInterest(img, maskReg):
    raw = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_color = (255,) * channel_count
    else:
        ignore_color = 255

    cv2.fillPoly(raw, maskReg, ignore_color)
    masked = cv2.bitwise_and(img, raw)
    return masked

# Applying DrawLines
def drawLinesP(img, HoughL, color = [255, 0, 0], thickness = 3):
    
    leftLaneLines = []
    leftLaneWeights = []
    rightLaneLines = []
    rightLaneWeights = []
    h1 = img.shape[0] * 0.65
    h2 = img.shape[0]
    
    for line in HoughL:
        for x1, y1, x2, y2 in line:
            y_diff = y2 - y1
            x_diff = x2 - x1
            slope = y_diff / x_diff
            intercept = y1 - slope * x1
            length = math.sqrt((y_diff ** 2) + (x_diff **2))
            
            if x1 == x2:
                continue

            if slope < 0 :
                leftLaneWeights.append(length)
                leftLaneLines.append((slope, intercept))
            
            else:
                rightLaneWeights.append(length)
                rightLaneLines.append((slope, intercept))

    leftLaneNorm = np.dot(leftLaneWeights, leftLaneLines) / np.sum(leftLaneWeights) if len(leftLaneWeights) > 0 else None
    rightLaneNorm = np.dot(rightLaneWeights, rightLaneLines) / np.sum(rightLaneWeights) if len(rightLaneWeights) > 0 else None

    if(rightLaneLines is not None):
        rslope, rintercept = rightLaneNorm
        y1, y2 = int(h1), int(h2)
        x1, x2 = int((y1 - rintercept) / rslope), int((y2 - rintercept) / rslope)

        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    if(leftLaneLines is not None):
        lslope, lintercept = leftLaneNorm
        y1, y2 = int(h1), int(h2)
        x1, x2 = int((y1 - lintercept) / lslope), int((y2 - lintercept) / lslope)

        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Applying Hough Transform
def applyHough(img, rho, theta, threshold, minLineLen, maxLineGap):
    HoughL = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLen, maxLineGap)
    print(HoughL)
    lineImg = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    drawLinesP(lineImg, HoughL)
    return lineImg

# Fetching Weighted Image
def fetchWeightedImage(original_img, line_img, a, b, g):
    return cv2.addWeighted(original_img, a, line_img, b, g)