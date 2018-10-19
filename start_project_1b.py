#
# Project 1, starter code part b
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt
import numpy 
from tensorflow import layers 
from tensorflow import initializers as init 
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.losses import get_regularization_loss


NUM_FEATURES = 8

learning_rate = 1e-7
epochs = 500
batch_size = 32
num_neuron = 30
seed = 10
np.random.seed(seed)

#read and divide data into test and train sets 
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
Y_data = np.asmatrix(Y_data).transpose()

idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data, Y_data = X_data[idx], Y_data[idx]

m = 3* X_data.shape[0] // 10
trainX, trainY = X_data[m:], Y_data[m:]

trainX = (trainX- np.mean(trainX, axis=0))/ np.std(trainX, axis=0)

# experiment with small datasets
trainX = trainX[:1000]
trainY = trainY[:1000]

# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, 1])

# Build the graph for the deep net
hidden_layer = layers.dense(x, 30, activation = tf.nn.relu, kernel_initializer = init.orthogonal(np.sqrt(2)), bias_initializer= init.zeros(), kernel_regularizer= l2_regularizer(1e-3))
output_layer = layers.dense(hidden_layer,1, kernel_initializer= init.orthogonal(), bias_initializer= init.zeros(),  kernel_regularizer= l2_regularizer(1e-3))

loss = tf.reduce_mean(tf.square(y_ - output_layer)) + get_regularization_loss()

#Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)
error = tf.reduce_mean(tf.square(y_ - output_layer))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	train_err = []
	for i in range(epochs):
		idm = 0
		while idm < 1000:
			nidx = idm + 32
			train_op.run(feed_dict={x: trainX[idm : nidx], y_: trainY[idm:nidx]})
			idm = nidx
		train_err.append(error.eval(feed_dict={x: trainX, y_: trainY}))

		if i % 100 == 0:
			print('iter %d: test error %g'%(i, train_err[i]))

# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_err)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train Error')
plt.show()
