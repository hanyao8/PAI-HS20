import numpy as np
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern, RBF

def custom_kernel1(x, x_prime):
    base_kernel = np.dot(x, x_prime.T)
    return base_kernel

def rbf_kernel(X1, X2, l=1.0, sigma_f=1.0):
    D_sq = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * D_sq)

def sklearn_best():
    #RBF(length_scale=np.exp(-1.18020928))+WhiteKernel(noise_level=np.exp(-5.85276903))
    #kernel = self.rbf_w*RBF(length_scale=self.rbf_ls)+WhiteKernel(noise_level=self.wk_nl)
    #kernel = RBF(length_scale=self.rbf_ls)+WhiteKernel(noise_level=self.wk_nl)
    return 1.0+0.5*Matern()+WhiteKernel()
