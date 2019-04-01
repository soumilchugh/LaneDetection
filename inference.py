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
net1 = conv2d(X, 16, 3, "Y0") #128
#net = tf.layers.batch_normalization(net)
print net1.shape
net2 = conv2d(net1, 32, 3, "Y1", strides=(2,2)) #128
#net = tf.layers.batch_normalization(net)
print net2.shape
net3 = conv2d(net2, 64, 3, "Y2", strides=(2, 2)) #64
#net = tf.layers.batch_normalization(net)
print net3.shape
net4 = conv2d(net3, 128, 3, "Y3", strides=(2, 2)) #32
print net4.shape
#net = tf.layers.batch_normalization(net)
net5 = conv2d(net4, 128, 3, "Y4", strides=(2, 2)) #32
print net5.shape
#net = tf.layers.batch_normalization(net)
#net6 = deconv2d(net5, 1, 23,40, 128, 128, "Y4_deconv") # 32
#net6 = tf.nn.relu(net6)
#print net6.shape
#net7 = tf.concat ([net6, net5], axis = -1)
#net8 = conv2d(net7, 128, 3, "Y5") #128
#net = tf.layers.batch_normalization(net)
#print net8.shape
net9 = deconv2d(net5, 1, 45,80, 128, 128, "Y3_deconv",strides=[1,2,2,1]) # 32
net9 = tf.nn.relu(net9)
net10 = tf.concat ([net9, net4], axis = -1)
#net = tf.layers.batch_normalization(net)
print net10.shape
net11 = conv2d(net10, 128, 3, "Y11") #1
net11 = deconv2d(net11, 1, 90,160, 64, 128, "Y2_deconv", strides=[1,2,2,1]) # 32
net11 = tf.nn.relu(net11)
net12 = tf.concat ([net11, net3], axis = -1)
net12 = conv2d(net12, 64, 3, "Y8") #128
print net12.shape
net13 = deconv2d(net12, 2, 180,320, 32, 64, "Y1_deconv", strides=[1, 2, 2, 1]) # 64
net13 = tf.nn.relu(net13)
#net = tf.layers.batch_normalization(net)
print net13.shape
net14 = tf.concat ([net13, net2], axis = -1)
net15 = conv2d(net14, 32, 3, "Y9") #1
print net15.shape
net16 = deconv2d(net15, 2, 360,640, 16, 32, "Y0_deconv", strides=[1, 2, 2, 1]) # 128
print net16.shape
net16 = tf.nn.relu(net16)
logits = deconv2d(net16, 1, 360,640, 1, 16, "logits_deconv") # 128

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
    trainData, trainLabels =  loadData()
    filenames = [img for img in glob.glob("/Users/soumilchugh/Downloads/assignment-3/good/ *.jpg")]
    for imagefilename in trainData:
        print imagefilename
        counter = counter + 1
        img = imread("/Users/soumilchugh/Downloads/assignment-3/validation-ground-truth/" + imagefilename)[:,:,:3]
        
        img = resize(img, (360, 640), mode='constant', preserve_range=True)
        X_test[0] = img
        test_image = X_test[0].astype(float)
        test_image = np.reshape(test_image, [-1, 360 , 640, 3])
        test_data = {X:test_image}
        test_mask = sess.run([logits],feed_dict=test_data)
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
        #test_mask = np.reshape(np.squeeze(test_mask), [360 , 640, 1])
        #for i in range(360):
        #   for j in range(640):
        #            if (sigmoid(test_mask[i][j])) >= 0.5:
        #                test_mask[i][j] = 255
        #            else:
        #                test_mask[i][j] = 0 
        #final =  test_mask.squeeze().astype(np.uint8)
        print counter
        #cv2.imwrite('/Users/soumilchugh/Downloads/assignment-3/masks/'+ imagefilename, final)





