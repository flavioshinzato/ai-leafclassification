import numpy as np
import pandas as pd
import cv2 as cv2
import os
import matplotlib
from PIL import Image, ImageDraw

def DTWDistance(s1, s2):
    DTW={}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1,-1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[i])
            DTW[(i,j)] = dist + min(DTW[(i-1,j)], DTW[(i, j-1)], DTW[(i-1, j-1)])

    return sqrt(DTW[len(s1)-1, len(s2)-1])

if __name__ == '__main__':
    img = []
    images = []
    for filename in os.listdir('/media/everlye/JARVIS/Faculdade/5 semestre/IA/Trab2/images'):
        img = cv2.imread(os.path.join('/media/everlye/JARVIS/Faculdade/5 semestre/IA/Trab2/images',filename), cv2.CV_LOAD_IMAGE_GRAYSCALE)
        thresh = 127
        maxValue = 255
        th, dst = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY)
        contours,hierarchy=cv2.findContours(dst, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dst, contours, -1,(255,255,255),3)

        if img is not None:
            images.append(dst)

        print images
    for i in range(len(images)):
        M = cv2.moments(images[i])
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])

        coord = []
        coord.append([centroid_x, centroid_y])
