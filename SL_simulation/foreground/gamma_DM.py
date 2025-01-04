import numpy as np
from scipy.stats import norm

def CDF_inv(F):
    mu_gamma = 1.25
    sigma = 0.2
    return norm.ppf(F, loc=mu_gamma, scale=sigma)