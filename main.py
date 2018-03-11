import numpy as np
import tensorflow as tf
from imageio import imwrite as ims
from tensorflow.examples.tutorials.mnist import input_data


def conv2d(x, kernel, out_channel, stride, name):
    with tf.variable_scope(name):
        in_channel = x.get_shape()[3]
        kernel_shape = [kernel, kernel, in_channel, out_channel]
        initializer = tf.truncated_normal_initializer(stddev=0.02)
        w = tf.get_variable('kernel', kernel_shape, initializer=initializer)

        bias_shape = [out_channel]
        initializer = tf.constant_initializer(0.0)
        b = tf.get_variable('bias', bias_shape, initializer=initializer)

        strides = [1, stride, stride, 1]
        return tf.nn.conv2d(x, w, strides=strides, padding='SAME') + b


def reverse_conv(x, conv_name, name):
    with tf.variable_scope(name):
        conv = tf.get_default_graph().get_operation_by_name(conv_name)
        strides = conv.get_attr('strides')
        batch_size = tf.shape(x)[0]
        output_shape = [batch_size] + conv.inputs[0].get_shape().as_list()[1:]
        kernel_shape = conv.inputs[1].get_shape().as_list()
        initializer = tf.truncated_normal_initializer(stddev=0.02)
        w = tf.get_variable('kernel', kernel_shape, initializer=initializer)
        return tf.nn.conv2d_transpose(
            x, w, output_shape=output_shape, strides=strides
        )


def reverse_reshape(x, reshape_name, name):
    with tf.name_scope(name):
        reshape = tf.get_default_graph().get_operation_by_name(reshape_name)
        output_shape = [-1] + reshape.inputs[0].get_shape().as_list()[1:]
        return tf.reshape(x, output_shape, name=name)


def get_output_shape(op_name):
    op = tf.get_default_graph().get_operation_by_name(op_name)
    return op.outputs[0].get_shape()


def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def dense(x, n_features, scope='Dense'):
    with tf.variable_scope(scope):
        input_feature = x.get_shape()[-1]
        kernel_shape = [input_feature, n_features]
        initializer = tf.random_normal_initializer(stddev=0.02)
        kernel = tf.get_variable(
            'kernel', kernel_shape, tf.float32, initializer=initializer)

        initializer = tf.constant_initializer(0.0)
        bias = tf.get_variable(
            'bias', [n_features], tf.float32, initializer=initializer)
        return tf.matmul(x, kernel) + bias


def _merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image[:, :, 0]
    return img


def encode(input_images, n_dim):
    # 28x28x1 -> 14x14x16
    h1 = lrelu(conv2d(input_images, 5, 16, stride=2, name='encoder_conv1'))
    # 14x14x16 -> 7x7x32
    h2 = lrelu(conv2d(h1, 5, 32, stride=2, name='encoder_conv2'))
    h2_flat = tf.reshape(
        h2, [-1, np.prod(h2.get_shape()[1:])], name='encoder_flatten')
    w_mean = dense(h2_flat, n_dim, 'z_mean')
    w_stddev = dense(h2_flat, n_dim, 'z_stddev')
    return w_mean, w_stddev


def sample(mean, stddev):
    samples = tf.random_normal(tf.shape(stddev), 0, 1, dtype=tf.float32)
    return mean + (stddev * samples)


def decode(z):
    shape = get_output_shape('encoder_flatten')
    z_develop = tf.nn.relu(dense(z, shape[1], scope='z_matrix'))
    z_matrix = reverse_reshape(
        z_develop, 'encoder_flatten', 'decoder_unflatten')
    h1 = tf.nn.relu(
        reverse_conv(z_matrix, 'encoder_conv2/Conv2D', 'decoder_conv2'))
    h2 = tf.nn.sigmoid(
        reverse_conv(h1, 'encoder_conv1/Conv2D', 'decoder_conv1'))
    return h2


def _mean_sum(var):
    return tf.reduce_sum(tf.reduce_mean(var, 0))


def cee(true, pred, name, offset=1e-8):
    with tf.name_scope(name):
        return -_mean_sum(
            true * tf.log(offset + pred) + (1-true) * tf.log(offset + 1 - pred)
        )


def kl_divergence(mean, stddev, name):
    with tf.name_scope(name):
        return _mean_sum(
            0.5 * (
                tf.square(mean) + tf.square(stddev) -
                tf.log(tf.square(stddev)) - 1
            )
        )


class VAE(object):
    def __init__(self, n_latent_dim=20):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        input_images = tf.placeholder(tf.float32, [None, 28, 28, 1])
        z_mean, z_stddev = encode(input_images, n_latent_dim)
        z_sample = sample(z_mean, z_stddev)
        gen_images = decode(z_sample)

        gen_loss = cee(input_images, gen_images, name='generative_loss')
        latent_loss = kl_divergence(z_mean, z_stddev, 'latent_loss')
        cost = gen_loss + latent_loss

        self.train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
        self.input_images = input_images
        self.generated_images = gen_images
        self.generation_loss = gen_loss
        self.latent_loss = latent_loss

    def train(self):
        visu = None
        saver = tf.train.Saver(max_to_keep=2)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter('results/summary', sess.graph)
        for epoch in range(10):
            for i in range(self.mnist.train.num_examples // 100):
                batch = self.mnist.train.next_batch(100)[0]
                batch = batch.reshape(-1, 28, 28, 1)
                if visu is None:
                    visu = batch

                sess.run(self.train_op, feed_dict={self.input_images: batch})
            gen_loss, lat_loss = sess.run(
                [self.generation_loss, self.latent_loss],
                feed_dict={self.input_images: batch}
            )
            print(
                'Epoch %d: Generative loss %8g Latent Loss %8g' %
                (epoch, np.mean(gen_loss), np.mean(lat_loss)))
            saver.save(
                sess, 'results/model',
                global_step=epoch
            )
            generated_test = sess.run(
                self.generated_images,
                feed_dict={self.input_images: visu}
            )
            ims('results/epoch_%05d.jpg' % epoch,
                np.hstack(
                    (
                        _merge(visu[:64], [8, 8]),
                        _merge(generated_test[:64], [8, 8]),
                    )
                )
            )


def _main():
    model = VAE()
    model.train()


if __name__ == '__main__':
    _main()
