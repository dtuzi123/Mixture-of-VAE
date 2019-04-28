# Mixture-of-VAE
This is  the implementation of the generative mixture VAE models

Abstract

Variational autoencoder (VAEs) are powerful latent variables models that can learn a probability distribution from the data. However, single VAEs has limitations on representation learning due its simple posterior and prior forms. In this paper, we proposed a new deep mixture learning framework based on idea of gaussian mixture model (GMM). However, to compare with classcal GMM that used shallow architecture and Expectation-Maximum (EM) algorithm which is not able to apply in the more complicated data distribution, we generalize gaussian mixture model into the deep learning framewrok with new evidence lower bound (called MELBO) where each component is implemented by an independent VAE that has its own inference mechanism and generation process. This design can allow mixture model to fast inference and to be easily trained using stochastic grdient decent. We further proposed to use the d variables Hilbert-Schmidt Independence Criterion (dHSIC) as a regularization term to enforce independence between encoder distributions, which encourage various components to learn data in different ways. We perform a series of experiments with different tasks to demonstrate that the proposed mixture model not only capture different aspects of data, but also discover more disentangled representations when comparing with single VAE. 


If you use the source code for you projects, please cite our paper.
