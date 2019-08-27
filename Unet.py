import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import csv
import cv2
import tensorflow as tf
import numpy as np
trainData = list();
trainLabels = list()
trainingLabels1 = list()
validData = list()
validLabels = list()
validationLabels = list()
learning_rate = 0.0005
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
def loadData():
    with open('/home/soumil/darknet/darknet-1/darknet/ground-truth/data.txt', 'rb') as f:
        trainData = list()
        trainLabels = list()
        reader = csv.reader(f)
        for row in reader:
            trainData.append(row[0])
            trainLabels.append('_mask_' + row[0])       

    return trainData, trainLabels

def loadValidData():
    with open('/home/soumil/darknet/darknet-1/darknet/validation-ground-truth/data.txt', 'rb') as f:
        validData = list()
        validLabels = list()
        reader = csv.reader(f)
        for row in reader:
            validData.append(row[0])
            validLabels.append('_mask_' + row[0])       
    return validData, validLabels

def convertOneHotTrain(trainTarget, trainTarget1):
    newtrain = np.zeros((trainTarget.shape[0], 2))
    for item in range(0, trainTarget.shape[0]):
        newtrain[item][0] = trainTarget[item]
        newtrain[item][1] = trainTarget1[item]
    return newtrain

def convertOneHotValid(validTarget, validTarget1):
    newvalid = np.zeros((validTarget.shape[0], 2))
    for item in range(0, validTarget.shape[0]):
        newvalid[item][0] = validTarget[item]
        newvalid[item][1] = validTarget1[item]
    return newvalid


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
def normalize(x):
    return (x.astype(float) - 128) / 128
def main():
    trainData, trainLabels = loadValidData()
    validData, validLabels = loadValidData()
    sess = tf.Session()
    initializer = tf.contrib.layers.xavier_initializer()
    with sess.as_default():
        X = tf.placeholder(tf.float32, [None, 360, 640,3], name='input')
        Y = tf.placeholder(tf.float32, [None, 360,640,1])
        net = conv2d(X, 16, 3, "Y0") 
        net1 = conv2d(net, 32, 3, "Y1",strides=(2, 2))
        net2 = conv2d(net1, 64, 3, "Y2", strides=(2, 2))
        net3 = conv2d(net2, 128, 3, "Y3", strides=(2, 2))
        net4 = conv2d(net3, 256, 3, "Y4", strides=(2, 2))
        net5 = deconv2d(net4, 1, 45,80, 128, 256, "Y4_deconv")
        net5 = tf.nn.relu(net5)
        concat1 = tf.concat([net5,net3],axis = 3)
        net6 = conv2d(concat1, 128, 3, "Y6")
        net7 = deconv2d(net6, 1, 90,160, 64, 128, "Y3_deconv")
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
        print (logits.shape)
        
        loss = tf.losses.sigmoid_cross_entropy(Y, logits)
        totalLoss = loss
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(totalLoss)
        init = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init)
        sess.run(init_l)
        saver = tf.train.Saver()
        iterationsList = list()
        trainingLossList = list()
        validationLossList = list()
        train_dataset = tf.data.Dataset.from_tensor_slices(np.array(trainData))
        test_dataset = tf.data.Dataset.from_tensor_slices(np.array(trainLabels))
        valid_dataset = tf.data.Dataset.from_tensor_slices(np.array(validData))
        validLabels_dataset = tf.data.Dataset.from_tensor_slices(np.array(validLabels))
        images = np.zeros((10, 360, 640, 3), dtype=np.uint8)
        labels = np.zeros((10, 360, 640, 1), dtype=np.bool)
        for epoch in range(10):
            print epoch
            combindedTrainDataset = tf.data.Dataset.zip((train_dataset, test_dataset)).shuffle(np.array(trainLabels).shape[0]).batch(10)
            iterator = combindedTrainDataset.make_initializable_iterator()
            next_element = iterator.get_next()
            sess.run(iterator.initializer)
            numberOfBatches = int(np.array(trainLabels).shape[0]/10)
            print numberOfBatches
            for i in range(numberOfBatches):
                val = sess.run(next_element)
                finaltrainingData = list()
                finalTraininglabels = list()
                for n,image in enumerate(val[0]):
                    img = imread('/home/soumil/darknet/darknet-1/darknet/validation-ground-truth/' +  image)[:,:,:3]
                    img = resize(img, (360, 640), mode='constant', preserve_range=True)
                    images[n] = img
                for n,image in enumerate(val[1]):
                    mask = np.zeros((360, 640, 1), dtype=np.bool)
                    mask_ = imread('/home/soumil/darknet/darknet-1/darknet/validation-ground-truth/' + image)
                    mask_ = np.expand_dims(resize(mask_, (360, 640), mode='constant', preserve_range=True), axis=-1)
                    mask = np.maximum(mask, mask_)
                    labels[n] = mask
                sess.run(optimizer, feed_dict={X:images,Y:labels})
                trainingLoss = sess.run(totalLoss, feed_dict={X:images,Y:labels})
                print "Training Loss is " +  str(trainingLoss) + "Step" + str(i)

            save_path = saver.save(sess, "/home/soumil/darknet/darknet-1/darknet/model/model.ckpt")
            print("Model saved in path: %s" % save_path)
            
        plt.figure()
        minimum = min(trainingLossList)
        print minimum
        plt.plot(iterationsList, trainingLossList, 'r')
        plt.plot(iterationsList, validationLossList, 'b')
        plt.gca().legend(('training Loss','validation Loss'))
        plt.savefig('_error_' + '.png')
        
if __name__ == '__main__':
    main()











