#
# Project 1, starter code part b
#
import tensorflow as tf
import numpy as np
import pylab as plt
from tensorflow import layers
from tensorflow import initializers as init
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.losses import get_regularization_loss

NUM_FEATURES = 8

NUM_FOLDS = 5
learning_rates = 1e-7
neurons = [20, 40, 60, 80, 100]
epochs = 500
batch_size = 32
num_neuron = 30
seed = 10
np.random.seed(seed)

# read and divide data into test and train sets
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:, :8], cal_housing[:, -1]
Y_data = np.asmatrix(Y_data).transpose()

indices = np.arange(X_data.shape[0])
np.random.shuffle(indices)
X_data = X_data[indices]
Y_data = Y_data[indices]

m = 3 * X_data.shape[0] // 10
testX, testY = X_data[:m], Y_data[:m]
trainX, trainY = X_data[m:], Y_data[m:]

# experiment with small datasets
trainX = trainX[:1000]
trainY = trainY[:1000]

for neuron_count in neurons:
    tf.reset_default_graph()
    with tf.Session() as sess:

        # Create the model
        lr = tf.placeholder(tf.float32, [])
        x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
        y_ = tf.placeholder(tf.float32, [None, 1])

        # Build the graph for the deep net
        hidden_layer = layers.dense(x, neuron_count, activation=tf.nn.relu, kernel_initializer=init.orthogonal(np.sqrt(2)),
                                    bias_initializer=init.zeros(), kernel_regularizer=l2_regularizer(1e-3))
        output_layer = layers.dense(hidden_layer, 1, kernel_initializer=init.orthogonal(), bias_initializer=init.zeros(),
                                    kernel_regularizer=l2_regularizer(1e-3))

        loss = tf.reduce_mean(tf.square(y_ - output_layer)
                              ) + get_regularization_loss()

        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_op = optimizer.minimize(loss)
        error = tf.reduce_mean(tf.square(y_ - output_layer))

        fold_errors = []
        for o in range(NUM_FOLDS):
            sess.run(tf.global_variables_initializer())

            xTrainX = np.split(trainX, NUM_FOLDS, axis=0)
            xTrainY = np.split(trainY, NUM_FOLDS, axis=0)

            xValidationX = xTrainX[o]
            xValidationY = xTrainY[o]
            xTrainX = np.concatenate(xTrainX[:o] + xTrainX[o + 1:])
            xTrainY = np.concatenate(xTrainY[:o] + xTrainY[o + 1:])

            mean = xTrainX.mean(axis=0)
            std = xTrainX.std(axis=0)

            xValidationX = (xValidationX - mean) / std
            xTrainX = (xTrainX - mean) / std

            val_err = []
            for i in range(epochs):
                idx = 0

                indices = np.arange(xTrainX.shape[0])
                np.random.shuffle(indices)

                xTrainX, xTrainY = xTrainX[indices], xTrainY[indices]
                while idx < xTrainX.shape[0]:
                    nidx = idx + 32
                    train_op.run(
                        feed_dict={x: xTrainX[idx: nidx], y_: xTrainY[idx:nidx], lr: learning_rates})
                    idx = nidx
                val_err.append(error.eval(
                    feed_dict={x: xValidationX, y_: xValidationY}))
            fold_errors.append(val_err)

        fold_errors = np.array(fold_errors)
        fold_errors = np.mean(fold_errors, axis=0)

        rand_idx = np.random.random()
        print('Average validation error: %g' % fold_errors[-1])
        plt.plot(range(100, epochs), fold_errors[100:])

plt.legend([str(x) for x in neurons])
plt.xlabel('Iterations')
plt.ylabel('Average Validation Error')
plt.yscale('log')
plt.show()
