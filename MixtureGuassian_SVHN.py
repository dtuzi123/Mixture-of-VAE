from __future__ import division
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
import math
from glob import glob
import scipy.io as sio
import tensorflow as tf
import numpy as np
from six.moves import xrange
from scipy.misc import imsave as ims
from keras.datasets import cifar10

from ops import *
from utils import *
from Fei_dataset import *
from Support import *

distributions = tf.distributions

d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')
d_bn4 = batch_norm(name='d_bn4')
'''
e_bn2 = batch_norm(name='e_bn2')
e_bn3 = batch_norm(name='e_bn3')
e_bn4 = batch_norm(name='e_bn4')
'''
g_bn0 = batch_norm(name='g_bn0')
g_bn1 = batch_norm(name='g_bn1')
g_bn2 = batch_norm(name='g_bn2')
g_bn3 = batch_norm(name='g_bn3')
g_bn4 = batch_norm(name='g_bn4')
g_bn5 = batch_norm(name='g_bn5')
g_bn6 = batch_norm(name='g_bn6')
g_bn7 = batch_norm(name='g_bn7')

batch_size = 16


def custom_layer(input_matrix, mix, resue=False):
    # with tf.variable_scope("custom_layer",reuse=resue):
    # w_init = tf.contrib.layers.variance_scaling_initializer()
    # b_init = tf.constant_initializer(0.)

    # weights = tf.get_variable(name="mix_weights", initializer=[0.25,0.25,0.25,0.25],trainable=True)
    weights = mix
    a1 = input_matrix[:, 0, :]
    a2 = input_matrix[:, 1, :]
    a3 = input_matrix[:, 2, :]
    a4 = input_matrix[:, 3, :]

    w1 = mix[:, 0:1]
    w2 = mix[:, 1:2]
    w3 = mix[:, 2:3]
    w4 = mix[:, 3:4]

    imageSize = 32
    a1 = tf.reshape(a1, (-1, imagesize * imagesize * 3))
    a2 = tf.reshape(a2, (-1, imagesize * imagesize * 3))
    a3 = tf.reshape(a3, (-1, imagesize * imagesize * 3))
    a4 = tf.reshape(a3, (-1, imagesize * imagesize * 3))

    outputs = w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4
    outputs = tf.reshape(outputs, (-1, imagesize, imagesize, 3))
    return outputs


def generator(z, batch_size=64, reuse=False):
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))
        '''
        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        h7 = deconv2d(h6, [batch_size, 32, 32, 64],
                      kernel, kernel, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        '''
        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8


def generator2(z, batch_size=64, reuse=False):
    with tf.variable_scope("generator2") as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))
        '''
        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        h7 = deconv2d(h6, [batch_size, 32, 32, 64],
                      kernel, kernel, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        '''
        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8


def generator3(z, batch_size=64, reuse=False):
    with tf.variable_scope("generator3") as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))
        '''
        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        h7 = deconv2d(h6, [batch_size, 32, 32, 64],
                      kernel, kernel, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        '''
        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8


def generator4(z, batch_size=64, reuse=False):
    with tf.variable_scope("generator4") as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))
        '''
        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        h7 = deconv2d(h6, [batch_size, 32, 32, 64],
                      kernel, kernel, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        '''
        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8


def encoder(image, batch_size=64, reuse=False):
    with tf.variable_scope("encoder") as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_mean = linear(h5, z_dim, 'e_mean')
        z_log_sigma_sq = linear(h5, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        z_mix = linear(h5, 1, 'e_mix')
        z_mix = tf.nn.softplus(z_mix)
        return (z_mean, z_log_sigma_sq, z_mix)


def encoder2(image, batch_size=64, reuse=False):
    with tf.variable_scope("encoder2") as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_mean = linear(h5, z_dim, 'e_mean')
        z_log_sigma_sq = linear(h5, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        z_mix = linear(h5, 1, 'e_mix')
        z_mix = tf.nn.softplus(z_mix)
        return (z_mean, z_log_sigma_sq, z_mix)


def encoder3(image, batch_size=64, reuse=False):
    with tf.variable_scope("encoder3") as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_mean = linear(h5, z_dim, 'e_mean')
        z_log_sigma_sq = linear(h5, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        z_mix = linear(h5, 1, 'e_mix')
        z_mix = tf.nn.softplus(z_mix)
        return (z_mean, z_log_sigma_sq, z_mix)


def encoder4(image, batch_size=64, reuse=False):
    with tf.variable_scope("encoder4") as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_mean = linear(h5, z_dim, 'e_mean')
        z_log_sigma_sq = linear(h5, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        z_mix = linear(h5, 1, 'e_mix')
        z_mix = tf.nn.softplus(z_mix)
        return (z_mean, z_log_sigma_sq, z_mix)


def HiddenOutputs(x_hat, x, dim_img, dim_z, n_hidden, keep_prob,last_term):
    mu1, sigma1, mix1 = encoder(x_hat, batch_size=64, reuse=True)
    mu2, sigma2, mix2 = encoder2(x_hat, batch_size=64, reuse=True)
    mu3, sigma3, mix3 = encoder3(x_hat, batch_size=64, reuse=True)
    mu4, sigma4, mix4 = encoder4(x_hat, batch_size=64, reuse=True)

    z1 = distributions.Normal(loc=mu1, scale=sigma1)
    z1_samples = z1.sample()

    z2 = distributions.Normal(loc=mu2, scale=sigma2)
    z2_samples = z2.sample()

    z3 = distributions.Normal(loc=mu3, scale=sigma3)
    z3_samples = z3.sample()

    z4 = distributions.Normal(loc=mu4, scale=sigma4)
    z4_samples = z4.sample()
    return z1_samples,z2_samples,z3_samples,z4_samples

def autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob,last_term):
    mu1, sigma1, mix1 = encoder(x_hat, batch_size=batch_size, reuse=False)
    mu2, sigma2, mix2 = encoder2(x_hat, batch_size=batch_size, reuse=False)
    mu3, sigma3, mix3 = encoder3(x_hat, batch_size=batch_size, reuse=False)
    mu4, sigma4, mix4 = encoder4(x_hat, batch_size=batch_size, reuse=False)

    z1 = distributions.Normal(loc=mu1, scale=sigma1)
    z1_samples = z1.sample()

    z2 = distributions.Normal(loc=mu2, scale=sigma2)
    z2_samples = z2.sample()

    z3 = distributions.Normal(loc=mu3, scale=sigma3)
    z3_samples = z3.sample()

    z4 = distributions.Normal(loc=mu4, scale=sigma4)
    z4_samples = z4.sample()

    y1 = generator(z1_samples, batch_size=batch_size, reuse=False)
    y2 = generator2(z2_samples, batch_size=batch_size, reuse=False)
    y3 = generator3(z3_samples, batch_size=batch_size, reuse=False)
    y4 = generator4(z4_samples, batch_size=batch_size, reuse=False)

    # mixing parameters
    y = tf.stack((y1, y2, y3, y4), axis=1)
    sum1 = mix1 + mix2 + mix3 + mix4
    mix1 = mix1 / sum1
    mix2 = mix2 / sum1
    mix3 = mix3 / sum1
    mix4 = mix4 / sum1
    mix = tf.concat([mix1, mix2, mix3, mix4], 1)
    mix_parameters = mix
    dist = tf.distributions.Dirichlet(mix)
    mix_samples = dist.sample()
    mix = mix_samples

    y = tf.stack((y1, y2, y3, y4), axis=1)
    outputs = custom_layer(y, mix)

    y = outputs

    m1 = np.zeros(dim_z, dtype=np.float32)
    m1[:] = 0
    v1 = np.zeros(dim_z, dtype=np.float32)
    v1[:] = 1

    # p_z1 = distributions.Normal(loc=np.zeros(dim_z, dtype=np.float32),
    #                           scale=np.ones(dim_z, dtype=np.float32))
    p_z1 = distributions.Normal(loc=m1,
                                scale=v1)

    m2 = np.zeros(dim_z, dtype=np.float32)
    m2[:] = 0
    v2 = np.zeros(dim_z, dtype=np.float32)
    v2[:] = 1
    p_z2 = distributions.Normal(loc=m2,
                                scale=v2)

    m3 = np.zeros(dim_z, dtype=np.float32)
    m3[:] = 0
    v3 = np.zeros(dim_z, dtype=np.float32)
    v3[:] = 1
    p_z3 = distributions.Normal(loc=m3,
                                scale=v3)

    m4 = np.zeros(dim_z, dtype=np.float32)
    m4[:] = 0
    v4 = np.zeros(dim_z, dtype=np.float32)
    v4[:] = 1
    p_z4 = distributions.Normal(loc=m4,
                                scale=v4)

    z = z1

    mu = mu1
    sigma = sigma1
    epsilon = 1e-8

    # additional loss
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - y), [1, 2, 3]))
    # kl_divergence = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + sigma - tf.square(mu) - tf.exp(sigma), 1))
    kl1 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z1, p_z1), 1))
    kl2 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z2, p_z2), 1))
    kl3 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z3, p_z3), 1))
    kl4 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z4, p_z4), 1))
    kl = kl1 + kl2 + kl3 + kl4
    kl_divergence = kl / 4.0

    # KL divergence between two Dirichlet distributions
    a1 = mix_parameters
    a2 = tf.constant((0.25, 0.25, 0.25, 0.25), shape=(batch_size, 4))

    r = tf.reduce_sum((a1 - a2) * (tf.polygamma(0.0, a1) - tf.polygamma(0.0, 1)), axis=1)
    a = tf.lgamma(tf.reduce_sum(a1, axis=1)) - tf.lgamma(tf.reduce_sum(a2, axis=1)) + tf.reduce_sum(tf.lgamma(a2),
                                                                                                    axis=-1) - tf.reduce_sum(
        tf.lgamma(a1), axis=1) + r
    kl = a
    kl = tf.reduce_mean(kl)

    p1 = 0.05

    loss = reconstruction_loss + kl_divergence * p1 + kl * p1 + last_term
    KL_divergence = kl_divergence
    marginal_likelihood = reconstruction_loss

    return y, z, loss, -marginal_likelihood, KL_divergence

def decoder_Components(x_hat, x, dim_img, dim_z, n_hidden, keep_prob,mix):
    mu1, sigma1, mix1 = encoder(x_hat, batch_size=batch_size, reuse=True)
    mu2, sigma2, mix2 = encoder2(x_hat, batch_size=batch_size, reuse=True)
    mu3, sigma3, mix3 = encoder3(x_hat, batch_size=batch_size, reuse=True)
    mu4, sigma4, mix4 = encoder4(x_hat, batch_size=batch_size, reuse=True)

    z1 = distributions.Normal(loc=mu1, scale=sigma1)
    z1_samples = z1.sample()

    z2 = distributions.Normal(loc=mu2, scale=sigma2)
    z2_samples = z2.sample()

    z3 = distributions.Normal(loc=mu3, scale=sigma3)
    z3_samples = z3.sample()

    z4 = distributions.Normal(loc=mu4, scale=sigma4)
    z4_samples = z4.sample()

    y1 = generator(z1_samples, batch_size=batch_size, reuse=True)
    y2 = generator2(z2_samples, batch_size=batch_size, reuse=True)
    y3 = generator3(z3_samples, batch_size=batch_size, reuse=True)
    y4 = generator4(z4_samples, batch_size=batch_size, reuse=True)

    y = tf.stack((y1, y2, y3, y4), axis=1)
    outputs = custom_layer(y, mix)
    return outputs


n_hidden = 500
IMAGE_SIZE_MNIST = 28
dim_img = IMAGE_SIZE_MNIST ** 2  # number of pixels for a MNIST image

myLatent_dim = 256
dim_z = myLatent_dim

# train
n_epochs = 10
batch_size = 64
learn_rate = 0.0001

# input placeholders

imagesize = 32
channel = 3
# In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
x_hat = tf.placeholder(tf.float32, shape=[None, imagesize, imagesize, channel], name='input_img')
x = tf.placeholder(tf.float32, shape=[None, imagesize, imagesize, channel], name='input_img')

image_dims = [imagesize, imagesize, channel]
x_hat = tf.placeholder(
    tf.float32, [batch_size] + image_dims, name='real_images')
mix = tf.placeholder(
    tf.float32, [batch_size,4])

x = x_hat
# dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# input for PMLR
z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')

last_term = tf.placeholder(tf.float32)
# network architecture
y, z, loss, neg_marginal_likelihood, KL_divergence = autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob,last_term)
z1_samples,z2_samples,z3_samples,z4_samples = HiddenOutputs(x_hat, x, dim_img, dim_z, n_hidden, keep_prob,last_term)
outputs_components = decoder_Components(x_hat, x, dim_img, dim_z, n_hidden, keep_prob,mix)

# optimization
train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

# train

min_tot_loss = 1e99
ADD_NOISE = False

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True  # 程序按需申请内

saver = tf.train.Saver(max_to_keep=4)
isWeight = True
best_likelihood = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob: 0.9})
    if isWeight:
        saver.restore(sess, 'F:/MixtureGaussian/AnthoerDataset/MixtureGaussian_SVHN_Weight2')
        # load dataset
        x_train, y_train, x_test, y_test = GetSVHN_DataSet()

        t1 = x_test[0:batch_size]
        count = np.shape(x_test)[0]
        imgs = []
        realImgs = []
        for i in range(count):
            x_fixed = x_test[i*batch_size:i*batch_size+batch_size]
            yy = sess.run(y,feed_dict={x_hat: x_fixed, keep_prob: 1})
            mix = np.zeros((batch_size,4))
            mix[:,0] = 1
            component1 = sess.run(outputs_components,feed_dict={x_hat: x_fixed, keep_prob: 1,mix:mix})

            for j in range(batch_size):
                imgs.append(yy[j])
                realImgs.append(x_fixed[j])
        imgs = np.array(imgs)
        realImgs = np.array(realImgs)

        msgarr = 0
        for i in range(np.shape(imgs)[0]):
            msg = np.square(realImgs[i],imgs[i])
            msgarr = msgarr + msg
        msgarr = msgarr / np.shape(imgs)[0]


        #yy = sess.run(y,feed_dict={x_hat: t1, x: t1, keep_prob: 1.0})
        #ims("results/" + "SVHN_Real" + str(0) + ".png", merge2(t1[:64], [8, 8]))
        #ims("results/" + "SVHN_Reco" + str(0) + ".png", merge2(yy[:64], [8, 8]))

        x_valid = x_train[50000:np.shape(x_train)[0]]
        y_valid = y_train[50000:np.shape(x_train)[0]]
        x_train = x_train[0:50000]
        y_train = y_train[0:50000]

        trainArray = []
        z1Arr = []
        z2Arr = []
        z3Arr = []
        z4Arr = []
        yArr = []

        count = np.shape(x_train)[0]
        count = 2000
        for i in range(int(count / batch_size)):
            x_fixed = x_train[i * batch_size:(i + 1) * batch_size]
            y_fixed = y_train[i * batch_size:(i + 1) * batch_size]
            #codes = sess.run(hiddenCodes, feed_dict={x_hat: x_fixed, keep_prob: 1})
            z1_samples_, z2_samples_, z3_samples_, z4_samples_ = sess.run(
                (z1_samples, z2_samples, z3_samples, z4_samples),
                feed_dict={x_hat: x_fixed, keep_prob: 1})

            for t in range(batch_size):
                #trainArray.append(codes[t])
                z1Arr.append(z1_samples_[t])
                z2Arr.append(z2_samples_[t])
                z3Arr.append(z3_samples_[t])
                z4Arr.append(z3_samples_[t])
                yArr.append(y_fixed[t])

        z1Arr = np.array(z1Arr)
        z2Arr = np.array(z2Arr)
        z3Arr = np.array(z3Arr)
        z4Arr = np.array(z4Arr)
        yArr = np.array(yArr)

        totalTrain = np.concatenate((z1Arr,z2Arr,z3Arr,z4Arr),axis=1)
        totalTrainY = yArr

        trainArray = []
        z1Arr = []
        z2Arr = []
        z3Arr = []
        z4Arr = []
        yArr = []
        count = np.shape(x_test)[0]

        for i in range(int(count / batch_size)):
            x_fixed = x_test[i * batch_size:(i + 1) * batch_size]
            y_fixed = y_test[i * batch_size:(i + 1) * batch_size]
            # codes = sess.run(hiddenCodes, feed_dict={x_hat: x_fixed, keep_prob: 1})
            z1_samples_, z2_samples_, z3_samples_, z4_samples_ = sess.run(
                (z1_samples, z2_samples, z3_samples, z4_samples),
                feed_dict={x_hat: x_fixed, keep_prob: 1})

            for t in range(batch_size):
                # trainArray.append(codes[t])
                z1Arr.append(z1_samples_[t])
                z2Arr.append(z2_samples_[t])
                z3Arr.append(z3_samples_[t])
                z4Arr.append(z3_samples_[t])
                yArr.append(y_fixed[t])

        z1Arr = np.array(z1Arr)
        z2Arr = np.array(z2Arr)
        z3Arr = np.array(z3Arr)
        z4Arr = np.array(z4Arr)
        yArr = np.array(yArr)

        totalTest = np.concatenate((z1Arr,z2Arr,z3Arr,z4Arr),axis=1)
        totalTestY = yArr

        trainArray = []
        z1Arr = []
        z2Arr = []
        z3Arr = []
        z4Arr = []
        yArr = []

        count = np.shape(x_valid)[0]
        for i in range(int(count / batch_size)):
            x_fixed = x_valid[i * batch_size:(i + 1) * batch_size]
            y_fixed = y_valid[i * batch_size:(i + 1) * batch_size]
            # codes = sess.run(hiddenCodes, feed_dict={x_hat: x_fixed, keep_prob: 1})
            z1_samples_, z2_samples_, z3_samples_, z4_samples_ = sess.run(
                (z1_samples, z2_samples, z3_samples, z4_samples),
                feed_dict={x_hat: x_fixed, keep_prob: 1})

            for t in range(batch_size):
                # trainArray.append(codes[t])
                z1Arr.append(z1_samples_[t])
                z2Arr.append(z2_samples_[t])
                z3Arr.append(z3_samples_[t])
                z4Arr.append(z3_samples_[t])
                yArr.append(y_fixed[t])

        z1Arr = np.array(z1Arr)
        z2Arr = np.array(z2Arr)
        z3Arr = np.array(z3Arr)
        z4Arr = np.array(z4Arr)
        yArr = np.array(yArr)

        totalValid = np.concatenate((z1Arr, z2Arr, z3Arr, z4Arr), axis=1)
        totalValidY = yArr

        totalTrainY = keras.utils.to_categorical(totalTrainY, 10)
        totalTestY = keras.utils.to_categorical(totalTestY, 10)
        totalValidY = keras.utils.to_categorical(totalValidY, 10)

        model = Sequential()
        model.add(Dense(1000, input_shape=(256*4,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        epochs = 10
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        history = model.fit(totalTrain, totalTrainY,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(totalValid,totalValidY))
        score = model.evaluate(totalTest, totalTestY,
                               batch_size=batch_size, verbose=1)

        bc = score

    # load dataset
    x_train, y_train, x_test, y_test = GetSVHN_DataSet()

    n_examples = np.shape(x_train)[0]
    total_batch = int(n_examples / batch_size)

    x_fixed = x_train[0:batch_size]

    for epoch in range(n_epochs):
        count = 0
        # Random shuffling

        # Loop over all batches
        for i in range(total_batch):

            # Compute the offset of the current minibatch in the data.
            batch_images = x_train[i * batch_size:(i + 1) * batch_size]
            batch_xs_input = batch_images
            batch_xs_target = batch_xs_input

            # add salt & pepper noise
            z1_samples_, z2_samples_, z3_samples_, z4_samples_ = sess.run(
                (z1_samples, z2_samples, z3_samples, z4_samples),
                feed_dict={x_hat: batch_xs_input, x: batch_xs_target, keep_prob: 0.9})

            b1, _ = hsic_gam(z1_samples_, z2_samples_)
            b2, _ = hsic_gam(z1_samples_, z3_samples_)
            b3, _ = hsic_gam(z1_samples_, z4_samples_)
            b4, _ = hsic_gam(z2_samples_, z3_samples_)
            b5, _ = hsic_gam(z2_samples_, z4_samples_)
            b6, _ = hsic_gam(z3_samples_, z4_samples_)
            lastvalue = b1 + b2 + b3 + b4 + b5 + b6

            # add salt & pepper noise

            if ADD_NOISE:
                batch_xs_input = batch_xs_input * np.random.randint(2, size=batch_xs_input.shape)
                batch_xs_input += np.random.randint(2, size=batch_xs_input.shape)

            _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                (train_op, loss, neg_marginal_likelihood, KL_divergence),
                feed_dict={x_hat: batch_xs_input, x: batch_xs_target, keep_prob: 0.9,last_term:lastvalue})

            test_loss_likelihood, test_loss_divergence = sess.run(
                ( neg_marginal_likelihood, KL_divergence),
                feed_dict={x_hat: x_test[0:batch_size], x: x_test[0:batch_size], keep_prob: 0.9})

            if test_loss_likelihood > best_likelihood:
                best_likelihood = test_loss_likelihood
                saver.save(sess, "F:/MixtureGaussian/AnthoerDataset/MixtureGaussian_SVHN_Weight")

            print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                epoch, tot_loss, loss_likelihood, loss_divergence))
        # print cost every epoch

        y_PRR = sess.run(y, feed_dict={x_hat: x_fixed, keep_prob: 1})
        y_RPR = np.reshape(y_PRR, (-1, 32, 32, 3))
        ims("results/" + "VAE" + str(epoch) + ".jpg", merge2(y_RPR[:64], [8, 8]))

        if epoch > 0:
            x_fixed_image = np.reshape(x_fixed, (-1, 32, 32, 3))
            ims("results/" + "Real" + str(epoch) + ".jpg", merge2(x_fixed_image[:64], [8, 8]))

    saver.save(sess, "F:/MixtureGaussian/AnthoerDataset/MixtureGaussian_SVHN_Weight2")


