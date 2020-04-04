import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_core.examples.tutorials.mnist import input_data

tf.disable_eager_execution()
tf.disable_control_flow_v2()


def plot_images(images, cls_true, cls_pred=None, title=None):
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap='binary')
        if cls_pred is None:
            ax_title = "True: {0}".format(cls_true[i])
        else:
            ax_title = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_title(ax_title)

        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        plt.suptitle(title, size=20)
    plt.show(block=False)


def plot_example_errors(images, cls_true, cls_pred, title=None):
    incorrect = np.logical_not(np.equal(cls_pred, cls_true))

    incorrect_images = images[incorrect]

    cls_pred = cls_pred[incorrect]
    cls_true = cls_true[incorrect]

    plot_images(images=incorrect_images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9],
                title=title)


class NN(object):
    def __init__(self):
        self.img_size_flat = 28 * 28
        self.n_classes = 10

        self.learning_rate = 0.001
        self.epochs = 10
        self.batch_size = 100
        self.display_freq = 100

        self.h1 = 256

        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.x_test = None
        self.y_test = None

    def load_data(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.x_train, self.y_train = mnist.train.images, mnist.train.labels
        self.x_valid, self.y_valid = mnist.validation.images, mnist.validation.labels
        self.x_test, self.y_test = mnist.test.images, mnist.test.labels

    @staticmethod
    def randomize(x, y):
        permutation = np.random.permutation(y.shape[0])
        shuffled_x = x[permutation, :]
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

    def fc_layer(self, x, num_units, name):
        with tf.variable_scope(name):
            in_dim = x.get_shape()[1]
            W = self.weight_variable(name, shape=[in_dim, num_units])
            b = self.bias_variable(name, shape=[num_units])
            layer = tf.matmul(x, W) + b
            return layer

    def run(self):
        x = tf.placeholder(tf.float32, shape=[None, self.img_size_flat])
        y = tf.placeholder(tf.float32, shape=[None, self.n_classes])

        fc1 = self.fc_layer(x, self.h1, 'fc1')
        output = self.fc_layer(fc1, self.n_classes, 'out')

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output), name='loss')
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='adam_op').minimize(loss)

        correct_prediction = tf.equal(tf.arg_max(output, 1), tf.arg_max(y, 1), name='correct_pred')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        prediction = tf.arg_max(output, 1, name='predictions')

        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
        sess.run(init)

        num_tr_iter = int(len(self.y_train) / self.batch_size)

        for epoch in range(self.epochs):
            print('Training epoch {}'.format(epoch + 1))
            self.x_train, self.y_train = self.randomize(self.x_train, self.y_train)
            for iteration in range(num_tr_iter):
                start = iteration * self.batch_size
                end = (iteration + 1) * self.batch_size
                x_batch, y_batch = self.get_next_batch(self.x_train, self.y_train, start, end)

                feed_dict_batch = {x: x_batch, y: y_batch}
                sess.run(optimizer, feed_dict=feed_dict_batch)

                if iteration % self.display_freq == 0:
                    loss_batch, acc_batch = sess.run([loss, accuracy], feed_dict=feed_dict_batch)
                    print("iter {0:}: \t loss={1:.3f} acc={2:.01%}".format(iteration, loss_batch, acc_batch))

            feed_dict_valid = {x: self.x_valid[:1000], y: self.y_valid[:1000]}
            loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
            print('---------------------------------------------------------')
            print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".format(epoch + 1, loss_valid, acc_valid))
            print('---------------------------------------------------------')


        feed_dict_test = {x: self.x_test[:1000], y: self.y_test[:1000]}
        loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
        print('---------------------------------------------------------')
        print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
        print('---------------------------------------------------------')

        cls_pred = sess.run(prediction, feed_dict=feed_dict_test)
        cls_true = np.argmax(self.y_test[:1000], axis=1)
        plot_images(self.x_test, cls_true, cls_pred, title='Correct Examples')
        plot_example_errors(self.x_test[:1000], cls_true, cls_pred, title='Misclassified Examples')
        plt.show()


def main():
        nn = NN()
        nn.load_data()
        nn.run()


if __name__ == '__main__':
    main()
