import tensorflow as tf
import mnist_data

import tensorflow.contrib.slim as slim
import time
import seaborn as sns
from utils import *
from scipy.misc import imsave as ims
from Assign_Dataset import *
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist
from Support import *
from Mnist_DataHandle import *

import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

distributions = tf.distributions

def custom_layer(input_matrix,mix,resue=False):
    #with tf.variable_scope("custom_layer",reuse=resue):
        #w_init = tf.contrib.layers.variance_scaling_initializer()
        #b_init = tf.constant_initializer(0.)

        #weights = tf.get_variable(name="mix_weights", initializer=[0.25,0.25,0.25,0.25],trainable=True)
        weights = mix
        a1 = input_matrix[:,0,:]
        a2 = input_matrix[:,1,:]
        a3 = input_matrix[:,2,:]
        a4 = input_matrix[:,3,:]

        w1 = mix[:,0:1]
        w2 = mix[:,1:2]
        w3 = mix[:,2:3]
        w4 = mix[:,3:4]
        outputs = w1*a1 + w2 *a2 + w3*a3+w4*a4
        return outputs

# Gaussian MLP as encoder
def gaussian_MLP_encoder2(x, n_hidden, n_output, keep_prob,resue = False):
    with tf.variable_scope("gaussian_MLP_encoder2",reuse=resue):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(x, w0) + b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.tanh(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output * 2 + 1], initializer=w_init)
        bo = tf.get_variable('bo', [n_output * 2 + 1], initializer=b_init)
        gaussian_params = tf.matmul(h1, wo) + bo

        # The mean parameter is unconstrained
        mean = gaussian_params[:, :n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:n_output * 2])
        mix = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output * 2:n_output * 2 + 1])
        return mean, stddev, mix

    return mean, stddev

def gaussian_MLP_encoder3(x, n_hidden, n_output, keep_prob,resue = False):
    with tf.variable_scope("gaussian_MLP_encoder3",reuse=resue):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(x, w0) + b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.tanh(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output * 2 + 1], initializer=w_init)
        bo = tf.get_variable('bo', [n_output * 2 + 1], initializer=b_init)
        gaussian_params = tf.matmul(h1, wo) + bo

        # The mean parameter is unconstrained
        mean = gaussian_params[:, :n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:n_output * 2])
        mix = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output * 2:n_output * 2 + 1])
        return mean, stddev, mix

    return mean, stddev

def gaussian_MLP_encoder1(x, n_hidden, n_output, keep_prob,resue = False):
    with tf.variable_scope("gaussian_MLP_encoder1", reuse=resue):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(x, w0) + b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.tanh(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output * 2 + 1], initializer=w_init)
        bo = tf.get_variable('bo', [n_output * 2 + 1], initializer=b_init)
        gaussian_params = tf.matmul(h1, wo) + bo

        # The mean parameter is unconstrained
        mean = gaussian_params[:, :n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:n_output * 2])
        mix = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output * 2:n_output * 2 + 1])
        return mean, stddev, mix

# Gaussian MLP as encoder
def gaussian_MLP_encoder(x, n_hidden, n_output, keep_prob,resue = False):
    with tf.variable_scope("gaussian_MLP_encoder",reuse=resue):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(x, w0) + b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.tanh(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output * 2 + 1], initializer=w_init)
        bo = tf.get_variable('bo', [n_output * 2 + 1], initializer=b_init)
        gaussian_params = tf.matmul(h1, wo) + bo

        # The mean parameter is unconstrained
        mean = gaussian_params[:, :n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:n_output*2])
        mix = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output*2:n_output*2+1])

    return mean, stddev,mix

# Bernoulli MLP as decoder
def bernoulli_MLP_decoder(z, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer-mean
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.sigmoid(tf.matmul(h1, wo) + bo)

    return y

def bernoulli_MLP_decoder1(z, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("bernoulli_MLP_decoder1", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer-mean
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.sigmoid(tf.matmul(h1, wo) + bo)

    return y

def bernoulli_MLP_decoder2(z, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("bernoulli_MLP_decoder2", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer-mean
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.sigmoid(tf.matmul(h1, wo) + bo)

    return y

def bernoulli_MLP_decoder3(z, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("bernoulli_MLP_decoder3", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer-mean
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.sigmoid(tf.matmul(h1, wo) + bo)

    return y

def Fixed_Gaussian(x_hat, x, dim_img, dim_z, n_hidden, keep_prob,mix_config):
    mu1, sigma1, mix1 = gaussian_MLP_encoder(x_hat, n_hidden, dim_z, keep_prob,resue=True)
    z1 = distributions.Normal(loc=mu1, scale=sigma1)

    mu2, sigma2, mix2 = gaussian_MLP_encoder1(x_hat, n_hidden, dim_z, keep_prob,resue=True)
    z2 = distributions.Normal(loc=mu2, scale=sigma2)

    mu3, sigma3, mix3 = gaussian_MLP_encoder2(x_hat, n_hidden, dim_z, keep_prob,resue=True)
    z3 = distributions.Normal(loc=mu3, scale=sigma3)

    mu4, sigma4, mix4 = gaussian_MLP_encoder3(x_hat, n_hidden, dim_z, keep_prob,resue=True)
    z4 = distributions.Normal(loc=mu4, scale=sigma4)

    '''
    for i in range(batch_size):
        mix1[i,:] = mix_config[0]
        mix2[i, :] = mix_config[1]
        mix3[i, :] = mix_config[2]
        mix4[i, :] = mix_config[3]
 
    sum1 = mix1 + mix2 + mix3 + mix4
    mix1 = mix1 / sum1
    mix2 = mix2 / sum1
    mix3 = mix3 / sum1
    mix4 = mix4 / sum1
    '''

    #mix = tf.concat([mix1, mix2, mix3, mix4], 1)
    mix2 = tf.constant((0.33,0.33 ,0.33, 0.0),shape=(batch_size,4))

    mix2 = tf.Session().run(mix2)
    mix2 = mix2[1]

    mix_parameters = mix2
    dist = tf.distributions.Dirichlet(mix2)
    mix_samples = dist.sample()
    mix = mix2

    z1_samples = z1.sample()
    z2_samples = z2.sample()
    z3_samples = z3.sample()
    z4_samples = z4.sample()

    # decoding
    y1 = bernoulli_MLP_decoder(z1_samples, n_hidden, dim_img, keep_prob,reuse=True)
    y1 = tf.clip_by_value(y1, 1e-8, 1 - 1e-8)

    y2 = bernoulli_MLP_decoder1(z2_samples, n_hidden, dim_img, keep_prob,reuse=True)
    y2 = tf.clip_by_value(y2, 1e-8, 1 - 1e-8)

    y3 = bernoulli_MLP_decoder2(z3_samples, n_hidden, dim_img, keep_prob,reuse=True)
    y3 = tf.clip_by_value(y3, 1e-8, 1 - 1e-8)

    y4 = bernoulli_MLP_decoder3(z4_samples, n_hidden, dim_img, keep_prob,reuse=True)
    y4 = tf.clip_by_value(y4, 1e-8, 1 - 1e-8)

    y = tf.stack((y1, y2, y3, y4), axis=1)
    mix = mix_config
    outputs = custom_layer(y, mix)
    return outputs

# Gateway
def autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob,last_term):

    # encoding
    mu1, sigma1, mix1 = gaussian_MLP_encoder(x_hat, n_hidden, dim_z, keep_prob)
    z1 = distributions.Normal(loc=mu1, scale=sigma1)

    mu2, sigma2,mix2 = gaussian_MLP_encoder1(x_hat, n_hidden, dim_z, keep_prob)
    z2 = distributions.Normal(loc=mu2, scale=sigma2)

    mu3, sigma3,mix3 = gaussian_MLP_encoder2(x_hat, n_hidden, dim_z, keep_prob)
    z3 = distributions.Normal(loc=mu3, scale=sigma3)

    mu4, sigma4,mix4 = gaussian_MLP_encoder3(x_hat, n_hidden, dim_z, keep_prob)
    z4 = distributions.Normal(loc=mu4, scale=sigma4)

    z1 = distributions.Normal(loc=mu1, scale=sigma1)
    z2 = distributions.Normal(loc=mu2, scale=sigma2)
    z3 = distributions.Normal(loc=mu3, scale=sigma3)
    z4 = distributions.Normal(loc=mu4, scale=sigma4)

    sum1 = mix1 + mix2 + mix3 + mix4
    mix1 = mix1 / sum1
    mix2 = mix2 / sum1
    mix3 = mix3 / sum1
    mix4 = mix4 / sum1
    mix = tf.concat([mix1, mix2,mix3,mix4],1)
    mix_parameters = mix
    dist = tf.distributions.Dirichlet(mix)
    mix_samples = dist.sample()
    mix = mix_samples

    # sampling by re-parameterization technique
    #z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    z1_samples = z1.sample()
    z2_samples = z2.sample()
    z3_samples = z3.sample()
    z4_samples = z4.sample()

    # decoding
    y1 = bernoulli_MLP_decoder(z1_samples, n_hidden, dim_img, keep_prob)
    y1 = tf.clip_by_value(y1, 1e-8, 1 - 1e-8)

    y2 = bernoulli_MLP_decoder1(z2_samples, n_hidden, dim_img, keep_prob)
    y2 = tf.clip_by_value(y2, 1e-8, 1 - 1e-8)

    y3 = bernoulli_MLP_decoder2(z3_samples, n_hidden, dim_img, keep_prob)
    y3 = tf.clip_by_value(y3, 1e-8, 1 - 1e-8)

    y4 = bernoulli_MLP_decoder3(z4_samples, n_hidden, dim_img, keep_prob)
    y4 = tf.clip_by_value(y4, 1e-8, 1 - 1e-8)

    y = tf.stack((y1,y2,y3,y4),axis=1)
    outputs = custom_layer(y,mix)

    y = outputs
    z = z1

    ttf = []
    ttf.append(z1_samples)
    ttf.append(z2_samples)
    ttf.append(z3_samples)
    ttf.append(z4_samples)

    dHSIC_Value = dHSIC(ttf)

    m1 = np.zeros(dim_z, dtype=np.float32)
    m1[:] = 0
    v1 = np.zeros(dim_z, dtype=np.float32)
    v1[:] = 1

    #p_z1 = distributions.Normal(loc=np.zeros(dim_z, dtype=np.float32),
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

    kl1 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z1, p_z1), 1))
    kl2 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z2, p_z2), 1))
    kl3 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z3, p_z3), 1))
    kl4 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z4, p_z4), 1))

    k1 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z1, z2), 1))
    k2 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z2, z3), 1))
    k3 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z3, z4), 1))
    k4 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z2, z3), 1))
    k5 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z2, z4), 1))
    k6 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z3, z4), 1))

    KL_divergence = (kl1 + kl2 + kl3 + kl4) / 4.0
    diverse_KL_divergence = last_term


    # loss
    marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)

    marginal_likelihood = tf.reduce_mean(marginal_likelihood)

    #KL divergence between two Dirichlet distributions
    a1 = mix_parameters
    a2 = tf.constant((0.25,0.25,0.25,0.25),shape=(batch_size,4))

    r = tf.reduce_sum((a1-a2)*(tf.polygamma(0.0,a1)-tf.polygamma(0.0,1)),axis=1)
    a = tf.lgamma(tf.reduce_sum(a1,axis=1)) - tf.lgamma(tf.reduce_sum(a2,axis=1)) +tf.reduce_sum(tf.lgamma(a2),axis=-1) - tf.reduce_sum(tf.lgamma(a1),axis=1) + r
    kl = a
    kl = tf.reduce_mean(kl)

    p1 = 1
    p2 = 1
    ELBO = marginal_likelihood - KL_divergence * p2

    loss = -ELBO + kl*p1 + diverse_KL_divergence + dHSIC_Value

    return y, z, loss, -marginal_likelihood, KL_divergence


def autoencoder2(x_hat, x, dim_img, dim_z, n_hidden, keep_prob):
    # encoding
    mu1, sigma1, mix1 = gaussian_MLP_encoder(x_hat, n_hidden, dim_z, keep_prob,resue=True)
    z1 = distributions.Normal(loc=mu1, scale=sigma1)

    mu2, sigma2, mix2 = gaussian_MLP_encoder1(x_hat, n_hidden, dim_z, keep_prob,resue=True)
    z2 = distributions.Normal(loc=mu2, scale=sigma2)

    mu3, sigma3, mix3 = gaussian_MLP_encoder2(x_hat, n_hidden, dim_z, keep_prob,resue=True)
    z3 = distributions.Normal(loc=mu3, scale=sigma3)

    mu4, sigma4, mix4 = gaussian_MLP_encoder3(x_hat, n_hidden, dim_z, keep_prob,resue=True)
    z4 = distributions.Normal(loc=mu4, scale=sigma4)

    z1 = distributions.Normal(loc=mu1, scale=sigma1)
    z2 = distributions.Normal(loc=mu2, scale=sigma2)
    z3 = distributions.Normal(loc=mu3, scale=sigma3)
    z4 = distributions.Normal(loc=mu4, scale=sigma4)

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

    # sampling by re-parameterization technique
    # z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    z1_samples = z1.sample()
    z2_samples = z2.sample()
    z3_samples = z3.sample()
    z4_samples = z4.sample()

    #performing inference tasks

    i = 0
    '''
    c2 = z1_samples[i, :] * mix[i, 0] + z2_samples[i, :] * mix[i, 1] + z3_samples[i, :] * mix[i, 2] + z4_samples[i, :] * \
         mix[i, 3]
    c2 = tf.reshape(c2, (1, 20))
    for i in range(batch_size):
        if i == 0:
            continue
        c1 = z1_samples[i,:]*mix[i,0]+z2_samples[i,:]*mix[i,1]+z3_samples[i,:]*mix[i,2]+z4_samples[i,:]*mix[i,3]
        c1 = tf.reshape(c1,(1,20))
        c2 = tf.concat([c2, c1],axis=0)
    z = c2
    '''
    z = tf.concat([z1_samples, z2_samples,z3_samples,z4_samples], axis=1)

    # decoding
    y1 = bernoulli_MLP_decoder(z1_samples, n_hidden, dim_img, keep_prob,reuse=True)
    y1 = tf.clip_by_value(y1, 1e-8, 1 - 1e-8)

    y2 = bernoulli_MLP_decoder1(z2_samples, n_hidden, dim_img, keep_prob,reuse=True)
    y2 = tf.clip_by_value(y2, 1e-8, 1 - 1e-8)

    y3 = bernoulli_MLP_decoder2(z3_samples, n_hidden, dim_img, keep_prob,reuse=True)
    y3 = tf.clip_by_value(y3, 1e-8, 1 - 1e-8)

    y4 = bernoulli_MLP_decoder3(z4_samples, n_hidden, dim_img, keep_prob,reuse=True)
    y4 = tf.clip_by_value(y4, 1e-8, 1 - 1e-8)

    y = tf.stack((y1, y2, y3, y4), axis=1)
    outputs = custom_layer(y, mix)

    y = outputs

    m1 = np.zeros(dim_z, dtype=np.float32)
    m1[:] = 0.4
    v1 = np.zeros(dim_z, dtype=np.float32)
    v1[:] = 2

    # p_z1 = distributions.Normal(loc=np.zeros(dim_z, dtype=np.float32),
    #                           scale=np.ones(dim_z, dtype=np.float32))
    p_z1 = distributions.Normal(loc=m1,
                                scale=v1)

    m2 = np.zeros(dim_z, dtype=np.float32)
    m2[:] = 0.5
    v2 = np.zeros(dim_z, dtype=np.float32)
    v2[:] = 1
    p_z2 = distributions.Normal(loc=m2,
                                scale=v2)

    m3 = np.zeros(dim_z, dtype=np.float32)
    m3[:] = -0.5
    v3 = np.zeros(dim_z, dtype=np.float32)
    v3[:] = 1
    p_z3 = distributions.Normal(loc=m3,
                                scale=v3)

    m4 = np.zeros(dim_z, dtype=np.float32)
    m4[:] = 0.6
    v4 = np.zeros(dim_z, dtype=np.float32)
    v4[:] = 0.8
    p_z4 = distributions.Normal(loc=m4,
                                scale=v4)

    kl1 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z1, p_z1), 1))
    kl2 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z2, p_z2), 1))
    kl3 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z3, p_z3), 1))
    kl4 = tf.reduce_mean(tf.reduce_sum(distributions.kl_divergence(z4, p_z4), 1))

    KL_divergence = (kl1 + kl2 + kl3 + kl4) / 4

    diverse = tf.reduce_mean(tf.reduce_sum(
        tf.square(z1_samples - z2_samples) + tf.square(z1_samples - z3_samples) + tf.square(
            z1_samples - z4_samples) + tf.square(z2_samples - z3_samples) + tf.square(z3_samples - z4_samples)))

    # loss
    marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)

    marginal_likelihood = tf.reduce_mean(marginal_likelihood)

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
    ELBO = marginal_likelihood - KL_divergence

    loss = -ELBO + kl * p1

    return y, z,mix, loss, -marginal_likelihood, KL_divergence

def MyOutputs(mix,z,dim_img, dim_z, n_hidden, keep_prob):
    z1_samples = z[:,0:20]
    z2_samples = z[:,20:40]
    z3_samples = z[:,40:60]
    z4_samples = z[:,60:80]

    y1 = bernoulli_MLP_decoder(z1_samples, n_hidden, dim_img, keep_prob, reuse=True)
    y1 = tf.clip_by_value(y1, 1e-8, 1 - 1e-8)

    y2 = bernoulli_MLP_decoder1(z2_samples, n_hidden, dim_img, keep_prob, reuse=True)
    y2 = tf.clip_by_value(y2, 1e-8, 1 - 1e-8)

    y3 = bernoulli_MLP_decoder2(z3_samples, n_hidden, dim_img, keep_prob, reuse=True)
    y3 = tf.clip_by_value(y3, 1e-8, 1 - 1e-8)

    y4 = bernoulli_MLP_decoder3(z4_samples, n_hidden, dim_img, keep_prob, reuse=True)
    y4 = tf.clip_by_value(y4, 1e-8, 1 - 1e-8)

    y = tf.stack((y1, y2, y3, y4), axis=1)
    outputs = custom_layer(y, mix)

    y = outputs
    return y

def decoder(z, dim_img, n_hidden):

    y = bernoulli_MLP_decoder(z, n_hidden, dim_img, 1.0, reuse=True)
    return y

def Outpiuts_Component(x_hat, x, dim_img, dim_z, n_hidden, keep_prob):
    # encoding
    mu1, sigma1, mix1 = gaussian_MLP_encoder(x_hat, n_hidden, dim_z, keep_prob, resue=True)
    z1 = distributions.Normal(loc=mu1, scale=sigma1)

    mu2, sigma2, mix2 = gaussian_MLP_encoder1(x_hat, n_hidden, dim_z, keep_prob, resue=True)
    z2 = distributions.Normal(loc=mu2, scale=sigma2)

    mu3, sigma3, mix3 = gaussian_MLP_encoder2(x_hat, n_hidden, dim_z, keep_prob, resue=True)
    z3 = distributions.Normal(loc=mu3, scale=sigma3)

    mu4, sigma4, mix4 = gaussian_MLP_encoder3(x_hat, n_hidden, dim_z, keep_prob, resue=True)
    z4 = distributions.Normal(loc=mu4, scale=sigma4)

    z1 = distributions.Normal(loc=mu1, scale=sigma1)
    z2 = distributions.Normal(loc=mu2, scale=sigma2)
    z3 = distributions.Normal(loc=mu3, scale=sigma3)
    z4 = distributions.Normal(loc=mu4, scale=sigma4)

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

    # sampling by re-parameterization technique
    # z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    z1_samples = z1.sample()
    z2_samples = z2.sample()
    z3_samples = z3.sample()
    z4_samples = z4.sample()

    # performing inference tasks

    i = 0
    '''
    c2 = z1_samples[i, :] * mix[i, 0] + z2_samples[i, :] * mix[i, 1] + z3_samples[i, :] * mix[i, 2] + z4_samples[i, :] * \
         mix[i, 3]
    c2 = tf.reshape(c2, (1, 20))
    for i in range(batch_size):
        if i == 0:
            continue
        c1 = z1_samples[i,:]*mix[i,0]+z2_samples[i,:]*mix[i,1]+z3_samples[i,:]*mix[i,2]+z4_samples[i,:]*mix[i,3]
        c1 = tf.reshape(c1,(1,20))
        c2 = tf.concat([c2, c1],axis=0)
    z = c2
    '''
    z = tf.concat([z1_samples, z2_samples, z3_samples, z4_samples], axis=1)

    # decoding
    y1 = bernoulli_MLP_decoder(z1_samples, n_hidden, dim_img, keep_prob, reuse=True)
    y1 = tf.clip_by_value(y1, 1e-8, 1 - 1e-8)

    y2 = bernoulli_MLP_decoder1(z2_samples, n_hidden, dim_img, keep_prob, reuse=True)
    y2 = tf.clip_by_value(y2, 1e-8, 1 - 1e-8)

    y3 = bernoulli_MLP_decoder2(z3_samples, n_hidden, dim_img, keep_prob, reuse=True)
    y3 = tf.clip_by_value(y3, 1e-8, 1 - 1e-8)

    y4 = bernoulli_MLP_decoder3(z4_samples, n_hidden, dim_img, keep_prob, reuse=True)
    y4 = tf.clip_by_value(y4, 1e-8, 1 - 1e-8)
    return y1,y2,y3,y4

def HiddenOuputs(x_hat, x, dim_img, dim_z, n_hidden, keep_prob,last_term):
    # encoding
    mu1, sigma1, mix1 = gaussian_MLP_encoder(x_hat, n_hidden, dim_z, keep_prob,True)
    z1 = distributions.Normal(loc=mu1, scale=sigma1)

    mu2, sigma2, mix2 = gaussian_MLP_encoder1(x_hat, n_hidden, dim_z, keep_prob,True)
    z2 = distributions.Normal(loc=mu2, scale=sigma2)

    mu3, sigma3, mix3 = gaussian_MLP_encoder2(x_hat, n_hidden, dim_z, keep_prob,True)
    z3 = distributions.Normal(loc=mu3, scale=sigma3)

    mu4, sigma4, mix4 = gaussian_MLP_encoder3(x_hat, n_hidden, dim_z, keep_prob,True)
    z4 = distributions.Normal(loc=mu4, scale=sigma4)

    z1 = distributions.Normal(loc=mu1, scale=sigma1)
    z2 = distributions.Normal(loc=mu2, scale=sigma2)
    z3 = distributions.Normal(loc=mu3, scale=sigma3)
    z4 = distributions.Normal(loc=mu4, scale=sigma4)

    z1_samples = z1.sample()
    z2_samples = z2.sample()
    z3_samples = z3.sample()
    z4_samples = z4.sample()

    z1_samples = tf.concat((mu1,sigma1),axis=1)
    z2_samples = tf.concat((mu2, sigma2), axis=1)
    z3_samples = tf.concat((mu3, sigma3), axis=1)
    z4_samples = tf.concat((mu4, sigma4), axis=1)

    return z1_samples,z2_samples,z3_samples,z4_samples

def Random_Generation(z_in):
    y1 = bernoulli_MLP_decoder(z_in, n_hidden, dim_img, keep_prob, reuse=True)
    y1 = tf.clip_by_value(y1, 1e-8, 1 - 1e-8)

    y2 = bernoulli_MLP_decoder1(z_in, n_hidden, dim_img, keep_prob, reuse=True)
    y2 = tf.clip_by_value(y2, 1e-8, 1 - 1e-8)

    y3 = bernoulli_MLP_decoder2(z_in, n_hidden, dim_img, keep_prob, reuse=True)
    y3 = tf.clip_by_value(y3, 1e-8, 1 - 1e-8)

    y4 = bernoulli_MLP_decoder3(z_in, n_hidden, dim_img, keep_prob, reuse=True)
    y4 = tf.clip_by_value(y4, 1e-8, 1 - 1e-8)

    return y1,y2,y3,y4

n_hidden = 500
IMAGE_SIZE_MNIST = 28
dim_img = IMAGE_SIZE_MNIST**2  # number of pixels for a MNIST image

dim_z = 50


# train
n_epochs = 100
batch_size = 128
learn_rate = 0.001

train_total_data, train_size, _, _, test_data, test_labels = mnist_data.prepare_MNIST_data()
n_samples = train_size
# input placeholders

# In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
x_hat = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
x = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img')

# dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# input for PMLR
z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')
last_term = tf.placeholder(tf.float32)

# network architecture
y, z, loss, neg_marginal_likelihood, KL_divergence = autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob,last_term)
z1_samples,z2_samples,z3_samples,z4_samples = HiddenOuputs(x_hat, x, dim_img, dim_z, n_hidden, keep_prob,last_term)
y1,y2,y3,y4 = Outpiuts_Component(x_hat, x, dim_img, dim_z, n_hidden, keep_prob)

# optimization
t_vars = tf.trainable_variables()
train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss,var_list=t_vars)

# train

total_batch = int(n_samples / batch_size)

min_tot_loss = 1e99
ADD_NOISE = False

train_data2_ = train_total_data[:, :-mnist_data.NUM_LABELS]
train_y = train_total_data[:, 784:784+mnist_data.NUM_LABELS]


# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_fixed = train_data2_[0:128]
saver = tf.train.Saver()

isWeight = False

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob: 0.9})
    if isWeight:
        saver.restore(sess, 'F:/MixtureGaussian/MixtureGaussian_DHSIC')

        m3 = np.zeros(dim_z, dtype=np.float32)
        m3[:] = -1
        v3 = np.zeros(dim_z, dtype=np.float32)
        v3[:] = 0
        p_z3 = distributions.Normal(loc=m3,
                                    scale=v3)
        z_samples = p_z3.sample(batch_size)

        z_samples1 = sess.run(z_samples)
        y1,y2,y3,y4 = Random_Generation(z_in)

        y1, y2, y3, y4 = sess.run(
            (y1, y2, y3, y4),
            feed_dict={z_in:z_samples1, keep_prob: 1.0})


        y1 = np.reshape(y1,(-1,28,28))
        y2 = np.reshape(y2,(-1,28,28))
        y3 = np.reshape(y3,(-1,28,28))
        y4 = np.reshape(y4,(-1,28,28))
        myNew = np.zeros((40,28,28))

        myNew[0:10,:,:] = y1[0:10,:,:]
        myNew[10:20,:,:] = y2[0:10,:,:]
        myNew[20:30,:,:] = y3[0:10,:,:]
        myNew[30:40,:,:] = y4[0:10,:,:]

        ims("results/" + "T" + str(0) + ".jpg", merge(myNew, [4, 10]))

        bc = 10
        '''
        ##reconstructed results
        x_test = np.reshape(x_test,(-1,28*28))
        batch = x_test[0:batch_size]
        yy = sess.run(
            y,
            feed_dict={x_hat: batch, x: batch, keep_prob: 0.9})

        y1 = sess.run(
            y1,
            feed_dict={x_hat: batch, x: batch, keep_prob: 0.9})
        y2 = sess.run(
            y2,
            feed_dict={x_hat: batch, x: batch, keep_prob: 0.9})
        y3 = sess.run(
            y3,
            feed_dict={x_hat: batch, x: batch, keep_prob: 0.9})
        y4 = sess.run(
            y4,
            feed_dict={x_hat: batch, x: batch, keep_prob: 0.9})

        y1=np.reshape(y1,(-1,28,28))
        y2 = np.reshape(y2, (-1, 28, 28))
        y3 = np.reshape(y3, (-1, 28, 28))
        y4 = np.reshape(y4, (-1, 28, 28))
        yy = np.reshape(yy,(-1,28,28))
        batch = np.reshape(batch,(-1,28,28))

        ims("results/" + "f" + str(0) + ".jpg", merge(y1[0:20], [1, 20]))
        ims("results/" + "f" + str(1) + ".jpg", merge(y2[0:20], [1, 20]))
        ims("results/" + "f" + str(2) + ".jpg", merge(y3[0:20], [1, 20]))
        ims("results/" + "f" + str(3) + ".jpg", merge(y4[0:20], [1, 20]))
        ims("results/" + "r" + str(0) + ".jpg", merge(batch[0:20], [1, 20]))
        ims("results/" + "t" + str(0) + ".jpg", merge(yy[0:20], [1, 20]))
        '''

        x_train = np.reshape(x_train,(-1,28*28))
        x_l,y_l = Split_dataset(x_train,y_train,100)
        x_l = x_train
        y_l = y_train
        z1_samples_, z2_samples_, z3_samples_, z4_samples_ = sess.run(
            (z1_samples, z2_samples, z3_samples, z4_samples),
            feed_dict={x_hat: x_l, x: x_l, keep_prob: 0.9})

        x_test = np.reshape(x_test,(-1,28*28))
        zArray = []
        count = np.shape(x_train)[0]
        for i in range(int(count/batch_size)):
            x_fixed = x_train[i*batch_size:(i+1)*batch_size]
            z1_samples_, z2_samples_, z3_samples_, z4_samples_ = sess.run(
                (z1_samples, z2_samples, z3_samples, z4_samples),
                feed_dict={x_hat: x_fixed, x: x_fixed, keep_prob: 0.9})
            for j in range(batch_size):
                zArray.append(z1_samples_[j])
        zArray = np.array(zArray)
        y_l = y_l[0:np.shape(zArray)[0]]

        zTestArray = []
        count = np.shape(x_test)[0]
        for i in range(int(count/batch_size)):
            x_fixed = x_test[i*batch_size:(i+1)*batch_size]
            z1_samples_, z2_samples_, z3_samples_, z4_samples_ = sess.run(
                (z1_samples, z2_samples, z3_samples, z4_samples),
                feed_dict={x_hat: x_fixed, x: x_fixed, keep_prob: 0.9})
            for j in range(batch_size):
                zTestArray.append(z1_samples_[j])
        zTestArray = np.array(zTestArray)
        y_test = y_test[0:np.shape(zTestArray)[0]]

        y_l = keras.utils.to_categorical(y_l, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        epochs = 10
        model = Sequential()
        model.add(Dense(100, input_shape=(dim_z*2,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        history = model.fit(zArray, y_l,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_split=0.1)
        score = model.evaluate(zTestArray, y_test,
                               batch_size=batch_size, verbose=1)

        from sklearn import svm
        model = svm.SVC(kernel='rbf', C=1, gamma=1)
        model.fit(zArray,y_l)
        s = model.score(zTestArray, y_test)

        mix_config = tf.placeholder(tf.float32, shape=[batch_size,4], name='input_weights2')

        mix_configValue = np.zeros((batch_size,4))
        index = 3
        mix_configValue[:,index] = 1.0

        for i in range(batch_size):
            a1 = np.random.rand(4)
            sum1 = a1[0] + a1[1] + a1[2] + a1[3]
            a1[0] = a1[0] / sum1
            a1[1] = a1[1] / sum1
            a1[2] = a1[2] / sum1
            a1[3] = a1[3] / sum1

            mix_configValue[i, 0] = a1[0]
            mix_configValue[i, 1] = a1[1]
            mix_configValue[i, 2] = a1[2]
            mix_configValue[i, 3] = a1[3]

            a1 = 0
            sum1 = 0.75
            mix_configValue[i, 0] = 0
            mix_configValue[i, 1] = 0
            mix_configValue[i, 2] = 1
            mix_configValue[i, 3] = 0

        outputs = Fixed_Gaussian(x_hat, x, dim_img, dim_z, n_hidden, keep_prob,mix_config)
        x_train = np.reshape(x_train, (-1, 784))

        tIndex = 10
        x_fixed = x_train[tIndex*batch_size:(tIndex+1)*batch_size]
        for j in range(batch_size):
            x_fixed[j, :] = x_train[17, :]

        y = sess.run(
            outputs,
            feed_dict={x_hat: x_fixed, x: x_fixed, keep_prob: 0.9,mix_config:mix_configValue})

        y_RPR = np.reshape(y, (-1, 28, 28))
        ims("results/" + "T" + str(0) + ".jpg", merge(y_RPR[:64], [8, 8]))

        x_fixed_image = np.reshape(x_fixed, (-1, 28, 28))
        ims("results/" + "Real" + str(0) + ".jpg", merge(x_fixed_image[:64], [8, 8]))

    else:

        n_epochs = 100
        for epoch in range(n_epochs):
            # Random shuffling
            np.random.shuffle(train_total_data)
            train_data_ = train_total_data[:, :-mnist_data.NUM_LABELS]

            # Loop over all batches
            for i in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (i * batch_size) % (n_samples)
                batch_xs_input = train_data_[offset:(offset + batch_size), :]
                batch_xs_target = batch_xs_input

                # add salt & pepper noise
                z1_samples_,z2_samples_,z3_samples_,z4_samples_ = sess.run(
                    (z1_samples,z2_samples,z3_samples,z4_samples),
                    feed_dict={x_hat: batch_xs_input, x: batch_xs_target, keep_prob: 0.9})

                b1,_ = hsic_gam(z1_samples_,z2_samples_)
                b2,_ = hsic_gam(z1_samples_,z3_samples_)
                b3,_ = hsic_gam(z1_samples_,z4_samples_)
                b4,_ = hsic_gam(z2_samples_,z3_samples_)
                b5,_ = hsic_gam(z2_samples_,z4_samples_)
                b6,_ = hsic_gam(z3_samples_,z4_samples_)
                lastvalue = b1+b2+b3+b4+b5+b6

                if ADD_NOISE:
                    batch_xs_input = batch_xs_input * np.random.randint(2, size=batch_xs_input.shape)
                    batch_xs_input += np.random.randint(2, size=batch_xs_input.shape)

                _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                    (train_op, loss, neg_marginal_likelihood, KL_divergence),
                    feed_dict={x_hat: batch_xs_input, x: batch_xs_target, keep_prob: 0.9,last_term:lastvalue})

            y_PRR = sess.run(y, feed_dict={x_hat: x_fixed, keep_prob: 1})
            y_RPR = np.reshape(y_PRR,(-1,28,28))
            ims("results/" + "VAE" + str(epoch) + ".jpg", merge(y_RPR[:64], [8, 8]))

            print("epoch %f: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
            epoch, tot_loss, loss_likelihood, loss_divergence))

            if epoch >0:
                x_fixed_image = np.reshape(x_fixed,(-1,28,28))
                ims("results/" + "Real" + str(epoch) + ".jpg", merge(x_fixed_image[:64], [8, 8]))

            # if minimum loss is updated or final epoch, plot results

    #        if min_tot_loss > tot_loss or epoch + 1 == n_epochs:
    #            min_tot_loss = tot_loss

    saver.save(sess, 'F:/MixtureGaussian/MixtureGaussian_DHSIC')