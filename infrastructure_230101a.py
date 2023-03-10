
# Generate a py file with useful functions.
# These functions are gradually being moved to the other py files.
# - match_rows_in_matrix
# - normalize_each_row_by_sum
# - tern2cart
# - similarity_matrix
# - sample_pool_sort

import pandas as pd
import numpy as np
import math

# Plotting tools
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import gpflow
# import GPy

from collections import namedtuple
import simpy
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import pdist, squareform
from gpflow.ci_utils import ci_niter
# import GPy
from scipy.spatial import distance as scipy_dist
from tabulate import tabulate
from sklearn.metrics.cluster import fowlkes_mallows_score as fmi
from sklearn.metrics.pairwise import pairwise_distances

# ------------- Infrastructure code ----------- 

def match_rows_in_matrix(M, R, dist = 0, eps = 1E-3):
    # find rows in matrix M that are near R within pairwise_distance of dist+eps 
    R = np.atleast_1d(R)
    if len(M.shape) == 1:
        M = M[:,None]
    if len(R.shape) == 1:
        R = R[:,None]
    d = pairwise_distances(M, R)
    a = d <= dist + eps
    match_ = a.sum(axis = 1).nonzero()[0]
    return match_

def normalize_each_row_by_sum(X):
    # normalize each row of a matrix by the row sum.
    # useful for composition data
    sumX = X.sum(axis = 1)[:,None]
    X = X / np.tile(sumX,(1, X.shape[1]) )
    return X

def similarity_matrix(X):
    # Compute similarity matrix W for XRD using cosine metric
    sigma = 1
    d_cos = squareform( pdist(X.squeeze(),'cosine') )
    W = np.exp(-(d_cos**2) / (2*sigma**2))
    return W

def tern2cart(tern_composition):
    # convert ternary data to cartesian coordinates.
    t = normalize_each_row_by_sum(tern_composition) * 100
    c = np.zeros((t.shape[0],2));
    c[:,1] = t[:,1] * np.sin(60 * np.pi/180)/100
    c[:,0] = t[:,0]/100 + c[:,1]*np.sin(30 * np.pi/180)/np.sin(60 * np.pi/180)
    return c
