import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_core.examples.tutorials.mnist import input_data
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.disable_control_flow_v2()
tf.disable_eager_execution()


def plot_images(original_images, noisy_images, reconstructed_images):
    num_images = original_images.shape[0]
    fig, axes = plt.subplots(num_images, 3, figsize=(9, 9))
    fig.subplots_adjust(hspace=.1, wspace=0)

    img_h = img_w = np.sqrt(original_images.shape[-1]).astype(int)
    for i, ax in enumerate(axes):
        ax[0].imshow(original_images[i].reshape((img_h, img_w)), cmap='gray')
        ax[1].imshow(noisy_images[i].reshape((img_h, img_w)), cmap='gray')
        ax[2].imshow(reconstructed_images[i].reshape((img_h, img_w)), cmap='gray')

        for sub_ax in ax:
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])

    for ax, col in zip(axes[0], ["Original Image", "Noisy Image", "Reconstructed Image"]):
        ax.set_title(col)

    fig.tight_layout()
    plt.show()


class Autoencoder(object):
    def __init__(self):
        self.logs_path = './logs/noiseRemovel'
        self.learning_rate = 0.001
        self.epochs = 1
        self.batch_size = 100
        self.img_size_flat = 28 * 28
        self.noise_level = 0.6
        self.display_freq = 100
        self.h1 = 256
        self.h2 = 128
        self.h3 = 256
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    def display_dataset_metadata(self):
        print("Training set: {}".format(len(self.mnist.train.labels)))
        print("Test set: {}".format(len(self.mnist.test.labels)))
        print("Validation set: {}".format(len(self.mnist.validation.labels)))

    def weight_variable(self, name, shape):
        initer = tf.truncated_normal_initializer(stddev=0.01)
        return tf.get_variable('W_' + name,
                               dtype=tf.float32,
                               shape=shape,
                               initializer=initer)

    def bias_variable(self, name, shape):
        initer = tf.constant(0., shape=shape, dtype=tf.float32)
        return tf.get_variable('b_' + name,
                               dtype=tf.float32,
                               initializer=initer)

    def fc_layer(self, x, num_units, name):
        with tf.variable_scope(name):
            in_dim = x.get_shape()[1]
            w = self.weight_variable(name, shape=[in_dim, num_units])
            b = self.bias_variable(name, shape=[num_units])
            layer = tf.matmul(x, w)
            layer += b
            layer = tf.nn.relu(layer)
            return layer

    def run(self):
        with tf.variable_scope('Input'):
            x_original = tf.placeholder(name='x_original', shape=[None, self.img_size_flat], dtype=tf.float32)
            x_noisy = tf.placeholder(name='x_noise', shape=[None, self.img_size_flat], dtype=tf.float32)

        fc1 = self.fc_layer(x_noisy, self.h1, 'layer_1')
        fc2 = self.fc_layer(fc1, self.h2, 'layer_2')
        fc3 = self.fc_layer(fc2, self.h3, 'layer_3')
        out = self.fc_layer(fc3, self.img_size_flat, 'output')

        with tf.variable_scope('Train'):
            with tf.variable_scope('Loss'):
                loss = tf.reduce_mean(tf.losses.mean_squared_error(x_original, out), name='loss')
            with tf.variable_scope('Train'):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='adam-op').minimize(loss)

        init = tf.global_variables_initializer()

        sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
        sess.run(init)

        num_tr_iter = int(self.mnist.train.num_examples / self.batch_size)

        x_valid_original = self.mnist.validation.images
        x_valid_noisy = x_valid_original + self.noise_level * np.random.normal(loc=0.0, scale=1.0, size=x_valid_original.shape)

        for epoch in range(self.epochs):
            print('Training epoch {}'.format(epoch + 1))
            for iteration in range(num_tr_iter):
                batch_x, _ = self.mnist.train.next_batch(self.batch_size)
                batch_x_noisy = batch_x + self.noise_level * np.random.normal(loc=0.0, scale=1.0, size=batch_x.shape)

                feed_dict_bath = {x_original: batch_x, x_noisy: batch_x_noisy}
                _ = sess.run([optimizer], feed_dict=feed_dict_bath)

                if not (iteration % self.display_freq):
                    loss_batch = sess.run(loss, feed_dict=feed_dict_bath)
                    print("iteration {0}:\t Reconstruction loss= {1:.3f}".format(iteration, loss_batch))

            feed_dict_bath = {x_original: x_valid_original, x_noisy: x_valid_noisy}
            loss_valid = sess.run(loss, feed_dict=feed_dict_bath)

            print('---------------------------------------------------------')
            print("Epoch: {0}, validation loss: {1:.3f}".format(epoch + 1, loss_valid))
            print('---------------------------------------------------------')

        num_test_samples = 5
        x_test = self.mnist.test.images[:num_test_samples]
        x_test_noisy = x_test + self.noise_level * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

        x_reconstruct = sess.run(out, feed_dict={x_noisy: x_test_noisy})
        loss_test = sess.run(loss, feed_dict={x_original: x_test, x_noisy: x_test_noisy})
        print('---------------------------------------------------------')
        print("Test loss of original image compared to reconstructed image : {0:.3f}".format(loss_test))
        print('---------------------------------------------------------')

        plot_images(x_test, x_test_noisy, x_reconstruct)
        sess.close()


def main():
    model = Autoencoder()
    model.display_dataset_metadata()
    model.run()


if __name__ == '__main__':
    main()
