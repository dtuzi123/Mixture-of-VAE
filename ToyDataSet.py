import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

distributions = tf.distributions
inta = 25

def Return_Samples(mu,sigmal,select):
    sum1 = 0
    i = select
    f = distributions.Normal(loc=mu[i], scale=sigmal[i])
    #for i in range(inta):
    #    f = distributions.Normal(loc=mu[i], scale=sigmal[i])
    #    sum1 = sum1 + f.sample()
    #sum1 = sum1 / inta
    sum1 = f.sample()
    return sum1

select = tf.placeholder(tf.int32)
mu = tf.placeholder(tf.float32, shape=[inta])
sigmal = tf.placeholder(tf.float32, shape=[inta])

samples = Return_Samples(mu,sigmal,select)


def CreateMixtureModels():
    minX= -3
    maxX = 3
    xArray = []
    yArray = []
    xxArray = []
    yyArray = []

    for i in range(4):
        xArray.append(-4)
        xxArray.append(-4 + i * 2)
    for i in range(4):
        xxArray.append(4)
        xArray.append(-4 + i * 2)
    for i in range(4):
        xArray.append(4)
        xxArray.append(-4 + i * 2)
    for i in range(3):
        xxArray.append(-4)
        xArray.append(-2 + i * 2)
    xArray.append(4)
    xxArray.append(4)

    for i in range(3):
        xArray.append(-2)
        xxArray.append(-2 + i * 2)
    for i in range(3):
        xArray.append(2)
        xxArray.append(-2 + i * 2)
    for i in range(3):
        xArray.append(0)
        xxArray.append(-2+i*2)

    for i in range(inta):
        yArray.append(0.01)
        yyArray.append(0.01)

    return xArray,yArray,xxArray,yyArray

x_mu,x_sigma,y_mu,y_sigma = CreateMixtureModels()

def Get_Dataset_Prior(trainCount=1000):
    data = np.zeros((trainCount, 2))
    indexArr = []
    mu2 = tf.placeholder(tf.float32)
    sigma2 = tf.placeholder(tf.float32)

    f = distributions.Normal(loc=0.0, scale=1.0)
    sample2 = f.sample()
    with tf.Session() as sess:
        for i in range(trainCount):
            xx = sess.run(sample2)
            yy = sess.run(sample2)
            data[i, 0] = xx
            data[i, 1] = yy
            indexArr.append(i)
        indexArr = np.array(indexArr)
    return data, indexArr

def Get_Dataset(trainCount=1000):
    data = np.zeros((trainCount, 2))
    indexArr = []
    with tf.Session() as sess:
        for i in range(trainCount):
            k = random.randint(0, inta - 1)
            xx = sess.run(samples, feed_dict={mu: x_mu, sigmal: x_sigma, select: k})
            yy = sess.run(samples, feed_dict={mu: y_mu, sigmal: y_sigma, select: k})
            data[i, 0] = xx
            data[i, 1] = yy
            indexArr.append(k)
        indexArr = np.array(indexArr)
    return data,indexArr

'''
with tf.Session() as sess:
    trainCount = 10000
    x,indexarray = Get_Dataset_Prior(trainCount)
    for i in range(trainCount):
        plt.scatter(x[i,0], x[i,1])

    plt.show()
'''



