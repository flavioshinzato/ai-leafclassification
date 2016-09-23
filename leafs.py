import numpy as np
import pandas as pd
import cv2 as cv2
import os
import matplotlib
from PIL import Image, ImageDraw
from math import sqrt

def DTWDistance(s1, s2):
    DTW={}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1,-1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = abs(s1[i] - s2[j])
            DTW[(i,j)] = dist + min(DTW[(i-1,j)], DTW[(i, j-1)], DTW[(i-1, j-1)])

    return DTW[len(s1)-1, len(s2)-1]


def p2pdistance(x1,y1,x2,y2):
    return sqrt((x2-x1)**2 + (y2-y1)**2)


if __name__ == '__main__':
    images_ts = []

    for filename in os.listdir('./images/'):
        img = cv2.imread(os.path.join('./images/',filename), cv2.CV_LOAD_IMAGE_GRAYSCALE)
        thresh = 127
        maxValue = 255
        th, dst = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY)
        contours, hierarchy=cv2.findContours(dst, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        M = cv2.moments(img)
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])
        print centroid_y, centroid_x

        ts = []

        for i in range(len(contours[0])):
            x = contours[0][i][0][0]
            y = contours[0][i][0][1]
            d = p2pdistance(x, y, centroid_x, centroid_y)
            ts.append(d)
            #print d
            #print ts

        images_ts.append(ts)