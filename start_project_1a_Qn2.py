#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt
import numpy as np
import matplotlib.pyplot as plt
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
num_neurons = 10
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

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_err_BS4 = []
    test_acc_BS4 = []
    train_err_BS8 = []
    test_acc_BS8 = []
    train_err_BS16 = []
    test_acc_BS16 = []
    train_err_BS32 = []
    test_acc_BS32 = []
    train_err_BS64 = []
    test_acc_BS64 = []
    timing_BS4 = []
    timing_BS8 = []
    timing_BS16 = []
    timing_BS32 = []
    timing_BS64 = []

    for i in range(epochs):
        start_time = datetime.now()
        idx = 0
        while idx < 1000:
            nidx = idx + 4
            train_op.run(
                feed_dict={x: trainX[idx: nidx], y_: trainY[idx:nidx]})
            idx = nidx
        timing_BS4.append((datetime.now() - start_time).total_seconds())
        train_err_BS4.append(loss.eval(feed_dict={x: trainX, y_: trainY}))
        test_acc_BS4.append(accuracy.eval(feed_dict={x: testX, y_: testY}))

        if i % 100 == 0:
            print('iter (BS4) %d: train err: %g test acc %g' % (
                i, train_err_BS4[i], test_acc_BS4[i]))

    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        start_time = datetime.now()
        idx = 0
        while idx < 1000:
            nidx = idx + 8
            train_op.run(
                feed_dict={x: trainX[idx: nidx], y_: trainY[idx:nidx]})
            idx = nidx

        timing_BS8.append((datetime.now() - start_time).total_seconds())
        train_err_BS8.append(loss.eval(feed_dict={x: trainX, y_: trainY}))
        test_acc_BS8.append(accuracy.eval(feed_dict={x: testX, y_: testY}))

        if i % 100 == 0:
            print('iter (BS8) %d: train err: %g test acc %g' % (
                i, train_err_BS8[i], test_acc_BS8[i]))

    sess.run(tf.global_variables_initializer())

    start_time = datetime.now()
    for i in range(epochs):
        start_time = datetime.now()
        idx = 0
        while idx < 1000:
            nidx = idx + 16
            train_op.run(
                feed_dict={x: trainX[idx: nidx], y_: trainY[idx:nidx]})
            idx = nidx

        timing_BS16.append((datetime.now() - start_time).total_seconds())
        train_err_BS16.append(loss.eval(feed_dict={x: trainX, y_: trainY}))
        test_acc_BS16.append(accuracy.eval(feed_dict={x: testX, y_: testY}))

        if i % 100 == 0:
            print('iter (BS16) %d: train err: %g test acc %g' % (
                i, train_err_BS16[i], test_acc_BS16[i]))

    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        start_time = datetime.now()
        idx = 0
        while idx < 1000:
            nidx = idx + 32
            train_op.run(
                feed_dict={x: trainX[idx: nidx], y_: trainY[idx:nidx]})
            idx = nidx

        timing_BS32.append((datetime.now() - start_time).total_seconds())
        train_err_BS32.append(loss.eval(feed_dict={x: trainX, y_: trainY}))
        test_acc_BS32.append(accuracy.eval(feed_dict={x: testX, y_: testY}))

        if i % 100 == 0:
            print('iter (BS32) %d: train err: %g test acc %g' % (
                i, train_err_BS32[i], test_acc_BS32[i]))

    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        start_time = datetime.now()
        idx = 0
        while idx < 1000:
            nidx = idx + 64
            train_op.run(
                feed_dict={x: trainX[idx: nidx], y_: trainY[idx:nidx]})
            idx = nidx

        timing_BS64.append((datetime.now() - start_time).total_seconds())
        train_err_BS64.append(loss.eval(feed_dict={x: trainX, y_: trainY}))
        test_acc_BS64.append(accuracy.eval(feed_dict={x: testX, y_: testY}))

        if i % 100 == 0:
            print('iter (BS64) %d: train err: %g test acc %g' % (
                i, train_err_BS64[i], test_acc_BS64[i]))


# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_err_BS4)
plt.plot(range(epochs), train_err_BS8)
plt.plot(range(epochs), train_err_BS16)
plt.plot(range(epochs), train_err_BS32)
plt.plot(range(epochs), train_err_BS64)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train error')
plt.legend(['Batch Size = 4', 'Batch Size = 8', 'Batch Size = 16',
            'Batch Size = 32', 'Batch Size = 64'], loc='best')
plt.show()

plt.figure(2)
plt.plot(range(epochs), test_acc_BS4)
plt.plot(range(epochs), test_acc_BS8)
plt.plot(range(epochs), test_acc_BS16)
plt.plot(range(epochs), test_acc_BS32)
plt.plot(range(epochs), test_acc_BS64)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Test accuracy')
plt.legend(['Batch Size = 4', 'Batch Size = 8', 'Batch Size = 16',
            'Batch Size = 32', 'Batch Size = 64'], loc='best')
plt.show()

plt.figure(3)
plt.plot(range(epochs), timing_BS4)
plt.plot(range(epochs), timing_BS8)
plt.plot(range(epochs), timing_BS16)
plt.plot(range(epochs), timing_BS32)
plt.plot(range(epochs), timing_BS64)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Timing Per Epoch')
plt.legend(['Batch Size = 4', 'Batch Size = 8', 'Batch Size = 16',
            'Batch Size = 32', 'Batch Size = 64'], loc='best')
plt.show()
