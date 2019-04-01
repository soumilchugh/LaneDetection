from __future__ import division
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from skimage.color import rgb2gray
import math
import csv
from skimage.measure import compare_ssim
from scipy.linalg import norm
from scipy import sum, average

trainData = list()
trainLabels = list()
accuracy = list()
counter = 0
def loadData():
    with open('/Users/soumilchugh/Downloads/assignment-3/validation-ground-truth/data.txt', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            trainData.append(row[0])
            trainLabels.append('_mask_' + row[0])       

    return trainData, trainLabels

def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

trainData, trainLabels = loadData()
IOU = list()

for n,imagefilename in enumerate(trainData):
    counter = counter + 1
    img = cv2.imread('/Users/soumilchugh/Downloads/assignment-3/validation-ground-truth/' + imagefilename,3)
    #print img.dtype
    maskImage = cv2.imread('/Users/soumilchugh/Downloads/assignment-3/masks/' + imagefilename,3)
    imagePixelCount = 0 ;
    totalPixelCount = 0
    lab = cv2.imread('/Users/soumilchugh/Downloads/assignment-3/validation-ground-truth/' + trainLabels[n],3)
    #label1 = cv2.cvtColor(maskImage, cv2.COLOR_BGR2GRAY)
    #label = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
    #label1 = cv2.Canny(label1,0,255)
    #label = cv2.Canny(labels,0,255)
    faceCoords = np.where(maskImage >= 128)
    maskImage[faceCoords[0],faceCoords[1],:] = [0,255,0]
    faceCoords = np.where(lab >= 128)
    lab[faceCoords[0],faceCoords[1],:] = [0,0,255]
    #segment = cv2.bitwise_or(img, maskImage)
    segment = cv2.bitwise_or(img, lab)

    print imagefilename

    #accuracy.append((label1Count/labelCount)*100)
    #print counter
    
    cv2.imwrite('/Users/soumilchugh/Downloads/assignment-3/results-original-image/ ' + imagefilename, segment)
    #cv2.imwrite('/Users/soumilchugh/Downloads/assignment-3/results/ ' + "label" + imagefilename, label)

    #cv2.imwrite('/Users/soumilchugh/Downloads/assignment-3/results/' +  "prediction" + imagefilename,label1) 



