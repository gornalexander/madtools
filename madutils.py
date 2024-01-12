import numpy as np

def calc_corr_factor(x, y):
    cov = np.cov(x, y)
    k = cov[0,1] / cov[0,0]
    return k