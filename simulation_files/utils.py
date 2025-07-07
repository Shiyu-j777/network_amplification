import numpy as np

def power_law_prob(N, decay_factor):
    """
    Generate a normalized power-law distribution with N discrete outcomes and decay factor k of x^{-k}
    Input:
        N (int): number of discrete outcomes
        decay_factor (float): the k (k>0) in the x^{-k}
    Output:
        prob_dist (list(float)): a np.array of length N that has the power-law distribution sum up to one
    """
    prob_dist = np.power(np.arange(1, N+1), -decay_factor)

    prob_dist /= np.sum(prob_dist)

    return prob_dist
