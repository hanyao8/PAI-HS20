import numpy as np
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern, RBF, RationalQuadratic

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
    #return 1.0+0.5*Matern()+WhiteKernel()
    return 0.386**2 + 0.165**2 *\
            Matern(length_scale=0.32, nu=1.5) +\
            WhiteKernel(noise_level=0.0026)

def sklearn_best2():
    #RBF(length_scale=np.exp(-1.18020928))+WhiteKernel(noise_level=np.exp(-5.85276903))
    #kernel = self.rbf_w*RBF(length_scale=self.rbf_ls)+WhiteKernel(noise_level=self.wk_nl)
    #kernel = RBF(length_scale=self.rbf_ls)+WhiteKernel(noise_level=self.wk_nl)
    #return 1.0+0.5*Matern()+WhiteKernel()
    return 0.386**2 + 0.165**2 *\
            Matern(length_scale=0.32, nu=1.5) +\
            Matern(length_scale=120, nu=1.5) +\
            WhiteKernel(noise_level=0.0026)

def sklearn_best3():
    return 0.33*RBF()+0.33*RationalQuadratic()+0.33*DotProduct()+WhiteKernel()

def sklearn_tunable():
    return 1.0*RBF()+1.0*Matern()+WhiteKernel()

def sklearn_tunable2():
    #return 0.33*RBF()+0.33*RationalQuadratic()+0.33*Matern()+WhiteKernel()
    return 0.232**2 * RBF(length_scale=0.662) + 0.0839**2 * RationalQuadratic(alpha=2.78e+04, length_scale=0.109) + 0.429**2 * Matern(length_scale=319, nu=1.5) + WhiteKernel(noise_level=0.0025)

def sklearn_tunable3():
    return 1.0+0.25*RBF()+0.25*RationalQuadratic()+0.25*DotProduct()+0.25*Matern()+WhiteKernel()

