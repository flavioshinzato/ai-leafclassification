# encoding=utf-8

import numpy as np
import pandas as pd
import cv2 as cv2
import os
import matplotlib
from PIL import Image, ImageDraw
from math import sqrt
import hcluster
from fastdtw import dtw

def p2pdistance(x1,y1,x2,y2):
    return sqrt((x2-x1)**2 + (y2-y1)**2)


if __name__ == '__main__':
    images_ts = []
    imlist = []

    for filename in os.listdir('./images/'):
        imlist.append(filename)
        img = cv2.imread(os.path.join('./images/',filename), cv2.CV_LOAD_IMAGE_GRAYSCALE)
        thresh = 127
        maxValue = 255
        th, dst = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY)
        contours, hierarchy=cv2.findContours(dst, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        M = cv2.moments(img)
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])

        ts = []

        for i in range(len(contours[0])):
            x = contours[0][i][0][0]
            y = contours[0][i][0][1]
            d = p2pdistance(x, y, centroid_x, centroid_y)
            ts.append(d)
            #print d
            #print ts

        images_ts.append(ts)

    print "Gerando a árvore"
    tree = hcluster.hcluster(images_ts, distance=dtw)

    print "Salvando a arvore"
    hcluster.drawdendrogram(tree, imlist, jpeg='saida.jpg')