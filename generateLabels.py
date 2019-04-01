from __future__ import division
import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

with open('/home/soumil/bdd-data-master/bdd100k/labels/bdd100k_labels_images_val.json') as f:
    data1 = json.load(f)
    finalList = list()
    finalDict = {}
    names = list()
    verticalCount = list()
    horizontalCount = list()
    verticalCounter = 0
    horizontalCounter = 0
    for data in data1:
        finalDict = {}
        verticalVertices = list()
        horizontalVertices = list()        
        finalDict['name'] = data['name']
        verticalCounter = 0
        for d in (data['labels']):
            for key in d.keys():
                finalVertices = list()
                if (key == 'category'):
                    if (d['category']) == 'lane':
                        vertices = list()
                        if d['attributes']['laneDirection'] == "parallel":
                            verticalCounter = verticalCounter + 1
                            for vertex in d['poly2d'][0]['vertices']:
                                finalVertices = [int(i/2) for i in vertex]
                                vertices.append(finalVertices)                                
                            horizontalVertices.append(vertices)
                        vertices = list()
                        if d['attributes']['laneDirection'] == "vertical":
                            verticalCounter = verticalCounter + 1
                            for vertex in d['poly2d'][0]['vertices']:
                                finalVertices = [int(i/2) for i in vertex]
                                vertices.append(finalVertices)
                            verticalVertices.append(vertices)
        finalDict['vertical'] = verticalVertices
        finalDict['parallel'] = horizontalVertices
        finalDict['verticalCount'] = verticalCounter
        finalList.append(finalDict)
    print np.array(finalList).shape
    print finalList[1]['name']
    print finalList[2]['name']
    print len(finalList[1]['parallel'])
    print len(finalList[1]['vertical'])
count = 0
with open('/home/soumil/darknet/darknet-1/darknet/validation-ground-truth/data.txt', 'w') as f:
    for index,final in enumerate(finalList):
        count += 1
        print final['name']
        f.write("%s," % final['name'])
        img = cv.imread('/home/soumil/bdd-data-master/bdd100k/images/100k/val/' + final['name'] , 3)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        resized_image1 = cv.resize(img, (0,0), fx=0.5, fy=0.5)
        resized_image = cv.resize(gray, (0,0), fx=0.5, fy=0.5)
        #resized_image = cv.resize(gray, (100, 100)) 
        #cv.imwrite('/home/soumil/darknet/darknet-1/darknet/originalImage/' + final['name'] ,resized_image) 
        mask = np.zeros_like(resized_image)
        for data in final['parallel']:
            maskvertices = list()
            for firstlist in data:
                maskvertices.append((tuple(firstlist)))
            #print maskvertices   
            polygons = np.array([maskvertices])
            cv.fillPoly(mask, polygons, 255)
        for data in final['vertical']:
            maskvertices = list()
            for firstlist in data:
                maskvertices.append((tuple(firstlist)))
            #print maskvertices   
            polygons = np.array([maskvertices])
            cv.fillPoly(mask, polygons, 255)
        #segment = cv.bitwise_or(resized_image, mask)
        print mask.dtype
        mask1 = cv.Canny(mask,0,255)
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(mask1, connectivity = 8)
        sizes = stats[1:,-1]
        print sizes
        counter = 0
        for i in range(nb_components-1):
            if (sizes[i] >= 100):
                counter = counter + 1
        print counter
        #f.write("%s\n" % nb_components)

        #mask4 = cv.Canny(mask,0,255)
        #mask2 = np.zeros_like(mask1)
        #lines = cv.HoughLinesP(mask1, 1, np.pi/180, 30, maxLineGap=250)
        #for line in lines:
        #    x1, y1, x2, y2 = line[0]
        #    cv.line(mask2, (x1, y1), (x2, y2), (255, 255, 255), 1)
        #mask1 = cv.Canny(mask2,0,255)

        #kernel = np.ones((5,5),np.uint8)
        #erosion = cv.erode(mask2,kernel,iterations = 1)
        #gauss = cv.medianBlur(mask2,5)
        #mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        cv.imwrite('/home/soumil/darknet/darknet-1/darknet/validation-ground-truth/' + final['name'], resized_image1) 
        cv.imwrite('/home/soumil/darknet/darknet-1/darknet/validation-ground-truth/' + '_mask_' + final['name'], mask1)
        if count == 2:
            break; 

    #verticalVertices = data['vertical']
    #parallelVertices = data['parallel']
    #if (len(verticalVertices)) == 2:
    #    polygons = np.array([[(289,596), (634,392)]])
    #cv.fillPoly(mask, polygons, 255)
    #cv.imwrite()
    #break;
