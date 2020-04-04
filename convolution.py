import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_core.examples.tutorials.mnist import input_data
from simple_neural_network import plot_images, plot_example_errors

tf.disable_eager_execution()
tf.disable_control_flow_v2()


class CNN(object):
    def __init__(self):
        self.img_size = 28
        self.img_size_flat = self.img_size * self.img_size
        self.n_classes = 10
        self.n_channels = 1

        self.learning_rate = 0.001
        self.epochs = 20
        self.batch_size = 100
        self.display_freq = 100

        self.filter_size1 = 5
        self.num_filters1 = 16
        self.stride1 = 1

        self.filter_size2 = 5
        self.num_filters2 = 32
        self.stride2 = 1

        self.h1 = 256

        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.x_test = None
        self.y_test = None

    def reformat(self, x):
        datasets = np.reshape(x, [-1, self.img_size, self.img_size, self.n_channels]).astype(np.float32)
        return datasets

    def load_data(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.x_train, self.y_train = mnist.train.images, mnist.train.labels
        self.x_train = self.reformat(self.x_train)

        self.x_valid, self.y_valid = mnist.validation.images, mnist.validation.labels
        self.x_valid = self.reformat(self.x_valid)

        self.x_test, self.y_test = mnist.test.images, mnist.test.labels
        self.x_test = self.reformat(self.x_test)

    @staticmethod
    def randomize(x, y):
        permutation = np.random.permutation(y.shape[0])
        shuffled_x = x[permutation, :, :, :]
        shuffled_y = y[permutation]
        return shuffled_x, shuffled_y

    @staticmethod
    def get_next_batch(x, y, start, end):
        x_batch = x[start: end]
        y_batch = y[start: end]
        return x_batch, y_batch

    @staticmethod
    def weight_variable(name, shape):
        initer = tf.truncated_normal_initializer(stddev=0.01)
        return tf.get_variable('W_' + name,
                               dtype=tf.float32,
                               shape=shape,
                               initializer=initer)

    @staticmethod
    def bias_variable(name, shape):
        initer = tf.constant(0., shape=shape, dtype=tf.float32)
        return tf.get_variable('b_' + name,
                               dtype=tf.float32,
                               initializer=initer)

    def fc_layer(self, x, num_units, name, use_relu=True):
        with tf.variable_scope(name):
            in_dim = x.get_shape()[1]
            W = self.weight_variable(name, shape=[in_dim, num_units])
            b = self.bias_variable(name, shape=[num_units])
            layer = tf.matmul(x, W) + b
            if use_relu:
                layer = tf.nn.relu(layer)
            return layer

    def conv_layer(self, x, filter_size, num_filters, stride, name):
        with tf.variable_scope(name):
            num_in_channel = x.get_shape().as_list()[-1]
            shape = [filter_size, filter_size, num_in_channel, num_filters]
            w = self.weight_variable(name, shape)
            b = self.bias_variable(name, [num_filters])
            layer = tf.nn.conv2d(x, w,
                                 strides=[1, stride, stride, 1],
                                 padding="SAME")
            layer += b
            layer = tf.nn.relu(layer)
            return layer

    def max_pool(self, x, ksize, stride, name):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding="SAME",
                              name=name)

    def flatten_layer(self, layer):
        with tf.variable_scope('Flatten_layer'):
            layer_shape = layer.get_shape()
            num_features = layer_shape[1: 4].num_elements()
            layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat

    def run(self):
        with tf.device('/gpu:0'):
            with tf.name_scope('Input'):
                x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, self.img_size, self.img_size, self.n_channels])
                y = tf.placeholder(name='y', dtype=tf.float32, shape=[None, self.n_classes])

            conv1 = self.conv_layer(x, self.filter_size1, self.num_filters1, self.stride1, 'conv1')
            pool1 = self.max_pool(conv1, ksize=2, stride=2, name='pool1')
            conv2 = self.conv_layer(pool1, self.filter_size2, self.num_filters2, self.stride2, 'conv2')
            pool2 = self.max_pool(conv2, ksize=2, stride=2, name='pool2')
            layer_flat = self.flatten_layer(pool2)
            fc1 = self.fc_layer(layer_flat, self.h1, 'FC1')
            output = self.fc_layer(fc1, self.n_classes, 'FC2', use_relu=False)

            print(x.get_shape())
            print(conv1.get_shape())
            print(pool1.get_shape())
            print(conv2.get_shape())
            print(pool2.get_shape())
            print(layer_flat.get_shape())
            print(fc1.get_shape())
            print(output.get_shape())

            with tf.variable_scope('Train'):
                with tf.variable_scope('Loss'):
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output), name='loss')
                with tf.variable_scope('Optimizer'):
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='adam-op').minimize(loss)
                with tf.variable_scope('Accuracy'):
                    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1), name='correct_prediction')
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
                with tf.variable_scope('Prediction'):
                    cls_prediction = tf.argmax(output, 1, name='prediction')

            init = tf.global_variables_initializer()

            sess = tf.InteractiveSession()
            sess.run(init)

            num_tr_writter = int(len(self.y_train) / self.batch_size)

            for epoch in range(self.epochs):
                print('Training epoch: {}'.format(epoch + 1))
                self.x_train, self.y_train = self.randomize(self.x_train, self.y_train)
                for iteration in range(num_tr_writter):
                    start = iteration * self.batch_size
                    end = (iteration + 1) * self.batch_size
                    x_batch, y_batch = self.get_next_batch(self.x_train, self.y_train, start, end)

                    feed_dict_batch = {x: x_batch, y: y_batch}
                    sess.run(optimizer, feed_dict=feed_dict_batch)

                    if iteration % self.display_freq == 0:
                        loss_batch, acc_batch = sess.run([loss, accuracy], feed_dict=feed_dict_batch)
                        print("iteration {0}:\t Reconstruction loss= {1:.3f}".format(iteration, loss_batch))

                feed_dict_valid = {x: self.x_valid, y: self.y_valid}
                loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
                print('---------------------------------------------------------')
                print(
                    "Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".format(epoch + 1, loss_valid,
                                                                                                 acc_valid))
                print('---------------------------------------------------------')

            feed_dict_test = {x: self.x_test, y: self.y_test}
            loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
            print('---------------------------------------------------------')
            print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
            print('---------------------------------------------------------')

            cls_pred = sess.run(cls_prediction, feed_dict=feed_dict_test)
            cls_true = np.argmax(self.y_test, axis=1)
            plot_images(self.x_test, cls_true, cls_pred, title='Correct Examples')
            plot_example_errors(self.x_test, cls_true, cls_pred, title='Misclassified Examples')
            plt.show()

            sess.close()


def main():
    cnn = CNN()
    cnn.load_data()
    cnn.run()


if __name__ == '__main__':
    main()
