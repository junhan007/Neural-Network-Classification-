#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt
import numpy as np
from tensorflow import layers
from tensorflow import initializers as init
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.losses import get_regularization_loss
from datetime import datetime


# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)


NUM_FEATURES = 36
NUM_CLASSES = 6

learning_rate = 0.01
epochs = 1000
batch_size = 32
seed = 10
np.random.seed(seed)

# read train data
train_input = np.loadtxt('sat_train.txt', delimiter=' ')
trainX, train_Y = train_input[:, :36], train_input[:, -1].astype(int)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))
train_Y[train_Y == 7] = 6

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1  # one hot matrix

# read test data
test_input = np.loadtxt('sat_test.txt', delimiter=' ')
testX, test_Y = test_input[:, :36], test_input[:, -1].astype(int)
testX = scale(testX, np.min(testX, axis=0), np.max(testX, axis=0))
test_Y[test_Y == 7] = 6

testY = np.zeros((test_Y.shape[0], NUM_CLASSES))
testY[np.arange(test_Y.shape[0]), test_Y-1] = 1  # one hot matrix


# experiment with small datasets
trainX = trainX[:1000]
trainY = trainY[:1000]

n = trainX.shape[0]

# 1 Hidden Layer
with tf.Session() as sess:
     # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # Build the graph for the deep net

    hidden_layer = layers.dense(x, 10, activation=tf.nn.relu, kernel_initializer=init.orthogonal(
        np.sqrt(2)), bias_initializer=init.zeros(), kernel_regularizer=l2_regularizer(1e-6))
    output_layer = layers.dense(hidden_layer, 6, kernel_initializer=init.orthogonal(
    ), bias_initializer=init.zeros(),  kernel_regularizer=l2_regularizer(1e-6))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y_, logits=output_layer)
    loss = tf.reduce_mean(cross_entropy) + get_regularization_loss()

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    correct_prediction = tf.cast(
        tf.equal(tf.argmax(output_layer, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    sess.run(tf.global_variables_initializer())
    train_acc = []
    test_acc = []

    for i in range(epochs):
        idx = 0
        while idx < 1000:
            nidx = idx + 32
            train_op.run(
                feed_dict={x: trainX[idx: nidx], y_: trainY[idx:nidx]})
            idx = nidx

        train_acc.append(accuracy.eval(feed_dict={x: trainX, y_: trainY}))
        test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))

        if i % 100 == 0:
            print('iter %d: train accuracy %g, test accuracy %g, 1 Hidden Layer' %
                  (i, train_acc[i], test_acc[i]))

# 2 Hidden Layer
with tf.Session() as sess:
     # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # Build the graph for the deep net

    hidden_layer = layers.dense(x, 10, activation=tf.nn.relu, kernel_initializer=init.orthogonal(
        np.sqrt(2)), bias_initializer=init.zeros(), kernel_regularizer=l2_regularizer(1e-6))
    hidden_layer1 = layers.dense(hidden_layer, 10, activation=tf.nn.relu, kernel_initializer=init.orthogonal(
        np.sqrt(2)), bias_initializer=init.zeros(), kernel_regularizer=l2_regularizer(1e-6))
    output_layer = layers.dense(hidden_layer1, 6, kernel_initializer=init.orthogonal(
    ), bias_initializer=init.zeros(),  kernel_regularizer=l2_regularizer(1e-6))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y_, logits=output_layer)
    loss = tf.reduce_mean(cross_entropy) + get_regularization_loss()

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    correct_prediction = tf.cast(
        tf.equal(tf.argmax(output_layer, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    sess.run(tf.global_variables_initializer())
    train_acc1 = []
    test_acc1 = []

    for i in range(epochs):
        idx = 0
        while idx < 1000:
            nidx = idx + 32
            train_op.run(
                feed_dict={x: trainX[idx: nidx], y_: trainY[idx:nidx]})
            idx = nidx

        train_acc1.append(accuracy.eval(feed_dict={x: trainX, y_: trainY}))
        test_acc1.append(accuracy.eval(feed_dict={x: testX, y_: testY}))

        if i % 100 == 0:
            print('iter %d: train accuracy %g, test accuracy %g, 2 Hidden Layer' %
                  (i, train_acc1[i], test_acc1[i]))

plt.figure(1)
plt.plot(range(epochs), train_acc)
plt.plot(range(epochs), test_acc)
plt.plot(range(epochs), train_acc1)
plt.plot(range(epochs), test_acc1)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Accuracy')
plt.legend(['Train Accuracy_1HL ', 'Test Accuracy_1HL',
            'Train Accuracy_2HL ', 'Test Accuracy_2HL'], loc='best')

plt.show(1)