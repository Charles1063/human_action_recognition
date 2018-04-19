# -*- coding: utf-8 -*-
import string
import itertools
import pickle
import matplotlib.pyplot as plt 

import tensorflow as tf

from PIL import Image
import cv2
import numpy as np

from sklearn import preprocessing
# 553 video per class, whole video set: 553x10

## read data
# read y file
objecty = []
with (open('/Users/wangsiyun/Documents/Image and video processing/Project/data/10/data_y10.p', 'rb')) as openfile:
    while True:
        try:
            objecty.append(pickle.load(openfile))
        except EOFError:
            break
data_y = np.asarray(objecty)
# data_y = data_y.T
print data_y.shape

# read x file
objectx = []
with (open("/Users/wangsiyun/Documents/Image and video processing/Project/data/10/data_x10.p", "rb")) as openfile:
    while True:
        try:
            objectx.append(pickle.load(openfile))
        except EOFError:
            break
data_x = np.asarray(objectx)
print data_x.shape

## 90% train and 10% validation
## all video: 5530, train: 4900(about 90%), validation: 630(about 10%)
shuf_ind1= np.random.permutation(data_x.shape[0])
x_train = data_x[shuf_ind1[0:4900],:]
y_train = data_y[shuf_ind1[0:4900],:]
validate_x = data_x[shuf_ind1[4900:],:]
validate_y = data_y[shuf_ind1[4900:],:]

n_features = 60*80*16
n_classes = 10
x = tf.placeholder(tf.float32, [None, n_features])
y_ = tf.placeholder(tf.float32, [None, n_classes])
images = tf.reshape(x, [-1, 60, 80, 16]) # down sample 60*80

training_iters = 8000

## initialize weight and bias
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.03)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.01, shape = shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'VALID')

def conv2d_st2(x,W):
	return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding = 'VALID')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

## build model
# conv1 and pool1
W_conv1 = weight_variable([6, 6, 16, 20])  # patch size 6x6, number of filters 20
b_conv1 = bias_variable([20])
conv1 = tf.nn.relu(conv2d_st2(images, W_conv1) + b_conv1) ## output size= 28*38*20
pool1 = max_pool_2x2(conv1)  # 14*19*20

# conv2 and pool2
W_conv2 = weight_variable([4, 5, 20, 32])  # patch size 4x5, number of filters 32
b_conv2 = bias_variable([32])
conv2 = tf.nn.relu(conv2d_st2(pool1, W_conv2) + b_conv2)  ## 6x8x32
pool2 = max_pool_2x2(conv2) ## 3x4x32

# fully connected layer
W_fc1 = weight_variable([3 * 4 * 32, 32]) 
b_fc1 = bias_variable([32])
pool2_flat = tf.reshape(pool2, [-1, 3 * 4 * 32])
fc1 = tf.nn.relu(tf.matmul(pool2_flat, W_fc1) + b_fc1)

# drop out
keep_prob = tf.placeholder('float')
fc1_drop = tf.nn.dropout(fc1, keep_prob)

# output layer
W_fc2 = weight_variable([32, 10])
b_fc2 = bias_variable([10])
y = tf.nn.softmax(tf.matmul(fc1_drop, W_fc2) + b_fc2)

## loss
# calculate cross entropy+ l2 regularization to avoid overfitting
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
reg_constant = 0.01
regularizers = (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
loss = cross_entropy + reg_constant * regularizers
## train
# leatning rate reduce method
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 200, 0.95, staircase=True)
# training method
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step)
# accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'))

# train begin
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
batch_size = 100
total_batch = int(x_train.shape[0] / batch_size)
n = []
acc = []
acc_v = []
acc_lr = []
acc_loss = []
for i in range(training_iters):
    shuf_ind= np.random.permutation(x_train.shape[0])
    for batch in range(total_batch):
        batch_x = x_train[shuf_ind[batch*batch_size : (batch+1)*batch_size], :]
        batch_y = y_train[shuf_ind[batch*batch_size : (batch+1)*batch_size], :]
    train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5}) #0.5~0.7
    # test accuracy for every 200 iterations
    if i%200 == 0:
    	train_accuracy = accuracy.eval(feed_dict={x:x_train[:4900,:], y_: y_train[:4900,:], keep_prob: 1.0})
    	print "step %d, training accuracy %g"%(i, train_accuracy)
    	n = np.append(n,i)
        acc = np.append(acc,train_accuracy)
        # learning_rate
        lr = learning_rate.eval()
        acc_lr = np.append(acc_lr,lr)
        # loss
        train_loss = loss.eval(feed_dict={x:x_train[:4900,:], y_: y_train[:4900,:], keep_prob: 1.0})
        acc_loss = np.append(acc_loss,train_loss)
    	# at the end of each epoch find the validation error
        validation_acc = accuracy.eval(feed_dict = {x: validate_x, y_: validate_y, keep_prob: 1.0})
        acc_v = np.append(acc_v,validation_acc)
        print "validate: step %d, training accuracy %g"%(i, validation_acc)

## plot
plt.figure(1)
plt.plot(n, acc, 'b*')
plt.plot(n,acc, 'r')
plt.plot(n,acc,'x-',label = 'Training accuracy')
plt.plot(n,acc_v,'+-',label = 'Validaion accuracy')
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.title('Accuracy')
plt.grid(True)
plt.legend(loc = 'best')

plt.figure(2)
plt.plot(n, acc_lr, 'b*')
plt.plot(n, acc_lr, 'r')
plt.title('learning rate')
plt.xlabel('iterations')
plt.ylabel('learning rate')

plt.figure(3)
plt.plot(n, acc_loss, 'b*')
plt.plot(n, acc_loss, 'r')
plt.title('loss')
plt.xlabel('iterations')
plt.ylabel('loss')

plt.show()

sess.close()

