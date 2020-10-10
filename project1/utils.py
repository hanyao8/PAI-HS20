import numpy as np

def make_real(np_obj,imag_thresh):
    #if (np.sum(np_obj)).imag > thresh:
    if np.max(np_obj.imag) > imag_thresh:
        print("Np object has significant imag component")
        raise(Exception)
    return np_obj.real

def pca_find_p(lam,pca_thresh):
    lam = make_real(lam,imag_thresh=1e-6)
    lam_frac_cumsum = np.cumsum(lam)/np.sum(lam)
    p = np.max([np.argmax(lam_frac_cumsum>pca_thresh),1])
    return p


