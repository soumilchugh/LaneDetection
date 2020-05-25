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
import glob
import os, argparse
from pathlib import Path

def loadData():
    with open('/Users/soumilchugh/Downloads/assignment-3/validation-ground-truth/data.txt', 'rb') as f:
        trainData = list()
        trainLabels = list()
        reader = csv.reader(f)
        for row in reader:
            trainData.append(row[0])
            trainLabels.append('_mask_' + row[0])       

    return trainData, trainLabels


def conv2d(input_tensor, depth, kernel, name, strides=(1, 1), padding="SAME"):
    return tf.layers.conv2d(input_tensor, filters=depth, kernel_size=kernel, strides=strides, padding=padding, activation=tf.nn.relu, name=name)

def deconv2d(input_tensor, filter_size, output_size,output_size1, out_channels, in_channels, name, strides = [1, 1, 1, 1]):
    dyn_input_shape = tf.shape(input_tensor)
    batch_size = dyn_input_shape[0]
    out_shape = tf.stack([batch_size, output_size, output_size1, out_channels])
    filter_shape = [filter_size, filter_size, out_channels, in_channels]
    w = tf.get_variable(name=name, shape=filter_shape)
    h1 = tf.nn.conv2d_transpose(input_tensor, w, out_shape, strides, padding='SAME')
    return h1

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, 360, 640,3], name='input')
Y = tf.placeholder(tf.float32, [None, 360,640,1])
net = conv2d(X, 16, 3, "Y0") 
net1 = conv2d(net, 32, 3, "Y1",strides=(2, 2))
net2 = conv2d(net1, 64, 3, "Y2", strides=(2, 2))
net3 = conv2d(net2, 128, 3, "Y3", strides=(2, 2))
net4 = conv2d(net3, 256, 3, "Y4", strides=(2, 2))
net5 = deconv2d(net4, 1, 45,80, 128, 256, "Y4_deconv",strides=[1, 2, 2, 1])
net5 = tf.nn.relu(net5)
concat1 = tf.concat([net5,net3],axis = 3)
net6 = conv2d(concat1, 128, 3, "Y6")
net7 = deconv2d(net6, 1, 90,160, 64, 128, "Y3_deconv",strides=[1, 2, 2, 1])
net7 = tf.nn.relu(net7)
concat2 = tf.concat([net7,net2],axis = 3)
net8 = conv2d(concat2, 64, 3, "Y7")
net9 = deconv2d(net8, 2, 180,320, 32, 64, "Y2_deconv", strides=[1, 2, 2, 1]) 
net9 = tf.nn.relu(net9)
concat3 = tf.concat([net9,net1],axis = 3)
net10 = conv2d(concat3, 32, 3, "Y8")
net11 = deconv2d(net10, 2, 360,640, 16, 32, "Y0_deconv", strides=[1, 2, 2, 1]) 
net11 = tf.nn.relu(net11)
concat4 = tf.concat([net11,net],axis = 3)
net12 = conv2d(concat4, 16, 3, "Y9")
logits = deconv2d(net12, 1, 360,640, 1, 16, "logits_deconv")

trainData, trainLabels = loadData()
init  = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "/Users/soumilchugh/Downloads/assignment-3/model/model.ckpt")
    graph = tf.get_default_graph()

    print("Model restored.")
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(graph, cmd='op', options=opts)

    counter  = 0
    X_test = np.zeros((1, 360, 640, 3), dtype=np.uint8)
    #trainData, trainLabels =  loadData()
    filepath = Path("C:/Users/soumi/Documents/test").glob('*.jpg')
    filenames = [file for file in filepath]

    for file in filenames:
        filename, file_extension = os.path.splitext(str(file))
        name = str(os.path.basename(filename));
        print (name)
        img = imread(str(file))[:,:,:3]
        img = resize(img, (360, 640), mode='constant', preserve_range=True)
        X_test[0] = img
        test_image = X_test[0].astype(float)
        test_image = np.reshape(test_image, [-1, 360 , 640, 3])
        test_data = {X:test_image}
        test_mask = sess.run([logits],feed_dict=test_data)
        test_mask = np.reshape(np.squeeze(test_mask), [360 , 640, 1])
        for i in range(360):
           for j in range(640):
                    if (sigmoid(test_mask[i][j])) >= 0.5:
                        test_mask[i][j] = 255
                    else:
                        test_mask[i][j] = 0 
        final =  test_mask.squeeze().astype(np.uint8)
        cv2.imwrite('/Documents/output/'+ name, final)
        
        # Feature Visualisation
        '''
        test = sess.run([net16],feed_dict=test_data)
        print np.array(test).shape
        test_mask1 = np.reshape(np.squeeze(test), [360 , 640, 16])
        #for k in range(16):
        test_mask2 = test_mask1[:,:,6]
        for i in range(360):
            for j in range(640):
                    test_mask2[i][j] = (sigmoid(test_mask2[i][j])*255)
        test_mask2 = test_mask2.squeeze().astype(np.uint8)
        cv2.imwrite(r'/Users/soumilchugh/Downloads/assignment-3/CNN-16/'+ 'Net' + imagefilename, test_mask2) 
        '''





