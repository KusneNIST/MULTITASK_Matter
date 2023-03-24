% A. Gilad Kusne, NIST, aaron.kusne@nist.gov, Release 3/1/2023
% If using this work for a publication, please cite:
% Kusne, A. Gilad, et al. "Scalable Multi-Agent Lab Framework for Lab Optimization" Matter 2023.

% Packages used:
% torch==1.11.0
% tensorflow==2.8.2
% tabulate==0.8.9
% simpy==4.0.1
% scipy==1.6.2
% scikit-learn==0.24.1
% pandas==1.2.4
% numpy==1.20.1
% matplotlib==3.3.4
% gpflow==2.2.1

% This software was developed by employees of the National Institute of
% Standards and Technology (NIST), an agency of the Federal Government and
% is being made available as a public service. Pursuant to title 17 United
% States Code Section 105, works of NIST employees are not subject to
% copyright protection in the United States.  This software may be subject
% to foreign copyright.  Permission in the United States and in foreign
% countries, to the extent that NIST may hold copyright, to use, copy,
% modify, create derivative works, and distribute this software and its
% documentation without fee is hereby granted on a non-exclusive basis,
% provided that this notice and disclaimer of warranty appears in all
% copies.

% THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER
% EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY
% WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED
% WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
% FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL
% CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR
% FREE.  IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT
% NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES,
% ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS
% SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR
% OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR
% OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF
% THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.

from infrastructure_230101a import match_rows_in_matrix, normalize_each_row_by_sum, tern2cart, similarity_matrix
from global_variables_and_monitors_230101a import performance_monitor

import pandas as pd
import numpy as np
import math

# Plotting tools
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import gpflow

from collections import namedtuple
import simpy
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import pdist, squareform
from gpflow.ci_utils import ci_niter
from scipy.spatial import distance as scipy_dist
from tabulate import tabulate
from sklearn.metrics.cluster import fowlkes_mallows_score as fmi
from scipy.special import gamma

# create class for synthesis instruments
# initialize instrument as a resource

# inst_params = {'Measure_Time':Measure_Time, 'Synth_Time':Synth_Time, 
#                'save_dir':r'/content/gdrive/My Drive/Research/jupyter/Networked ML/gpyflow_model',
#                'meas_type':meas_type,
#                'sample_pool_manager':spm_i}

class instrument_synthesis:
    def __init__(self, env, params, unique_ID):
        self.unique_ID = unique_ID
        self.env = env
        self.sample_synthesis = simpy.Resource(env, capacity = 1) # can only work on one job at a time.
        self.Synth_Time = params['Synth_Time']
        self.curr_synth_list = None
        self.spm = params['sample_pool_manager']
        
    def synthesis(self, target_composition):
        # function called to synthesize a sample at composition target_composition and add it to the sample_pool.
        if type(target_composition) is np.ndarray:
            target_composition = target_composition.item()
        #
        self.spm.synth_list_add(target_composition)
        # system yields to environment for amount of time Synth_Time and then returns
        # assume when it returns we have the sample
        
        yield self.env.timeout(self.Synth_Time)
        #     print('finished synth')
        max_item_index = self.spm.get_sample_index_max()
        # print('synth, max item in sample pool:', max_item_index)
        # add the new sample to the sample_pool.
        new_sample_index = max_item_index
        Sample = namedtuple('Sample', 'sample_index, composition, structure, funcprop, status')
        self.spm.sample_pool_append(Sample(new_sample_index, target_composition, 0, 0, 'in'))
        self.spm.synth_list_remove(target_composition)

        print(f'synth {str(self.unique_ID)[:5]} synthesized sample c:{target_composition} at {self.env.now}.')
        self.spm.all_synthesized.append(target_composition) # add new synthesized to list of all synthesized.
        # print(f'put {target_composition} in pool')
        return new_sample_index # return the index for the new sample.

class instrument_measure:
    # create class for measurement instruments
    # set up each measurement instrument as a resource that can only work on one job at a time.
    def __init__(self, env, focus, params, unique_ID):
        self.focus = focus
        self.unique_ID = unique_ID
        self.env = env
        self.sample_measure = simpy.Resource(env, capacity = 1) # can only work on one job at a time.
        self.Measure_Time = params['Measure_Time']
        self.meas_type = params['meas_type'] #'sim'
        self.save_dir = params['save_dir']
        self.spm = params['sample_pool_manager']
        if self.meas_type == 'sim_Raman_real' or self.meas_type == 'sim_Raman_hard':
            self.struct_data = np.load(self.save_dir + '/raman.npz')
            if self.meas_type == 'sim_Raman_real':
                self.func_prop = tf.saved_model.load(self.save_dir)
            
    def measure(self, target_composition):
        # function called to request the measurement of sample index target_index in the sample_pool
        # returns the measurement value.
        if type(target_composition) is np.ndarray:
            target_composition = target_composition.item()
        
        get0 = self.spm.list_sample_pool_items()
        sample = yield self.env.process(self.spm.get_sample(target_composition))
        get1 = self.spm.list_sample_pool_items()
        
        print(f'meas-{self.focus} {str(self.unique_ID)[:5]} got c:{sample.composition} at {self.env.now}')
        
        self.spm.lent_list_add(target_composition, self.focus)
        
        # if we're requesting a mag measurement, then timeout for Measure_Time and return the measurement.
        if self.focus == 'funcprop': 
            yield self.env.timeout(self.Measure_Time)
            measurement_results = self.measure_funcprop(sample.composition)
        # if we're requesting an xrd measurement, then timeout for Measure_Time and return the measurement.
        elif self.focus == 'structure':
            # print(f'structure measuring {sample.sample_index}')
            yield self.env.timeout(self.Measure_Time)
            measurement_results = self.measure_structure(sample.composition)
        
        put0 = self.spm.list_sample_pool_items()
        yield self.env.process(self.spm.put_sample(sample))
        put1 = self.spm.list_sample_pool_items()
        
        sget0 = sample.composition in np.asarray([s.composition for s in get0])
        sget1 = sample.composition in np.asarray([s.composition for s in get1])
        sput0 = sample.composition in np.asarray([s.composition for s in put0])
        sput1 = sample.composition in np.asarray([s.composition for s in put1])
        
        print(f'>>>{self.focus} {repr(self.unique_ID)[:5]} want c:{target_composition}, got c:{sample.composition} pre/post get:{sget0}/{sget1}, put:{sput0}/{sput1}')
        
        # !!!!!!!!!!!!! SHOULD THIS BE COMPOSITION BASED???
        self.spm.lent_list_remove(target_composition)
        print(f'meas-{self.focus} {str(self.unique_ID)[:5]} released c:{sample.composition} at {self.env.now}')
        yield self.env.timeout(1)
        return measurement_results # return the measurement
   
    def measure_funcprop(self, X):
        # function for simulating measuring functional property
        X = np.atleast_1d(X)
        if self.meas_type == 'sim':
            y = np.exp(-1*((X-0.05)**2)/(5E-3)) + .1*np.exp(-1*((X-1)**2)/(1)) + np.random.normal(0., .01, X.shape)
        elif self.meas_type == 'sim_Raman_hard':
            # split at 8 and 14
            # Current paper figure for hard challenge
            yg0 = .2*np.exp(-.5*(X-5.)**2 / 2.)
            yg0[X > 8.] = 0
            yg1 = .2*np.exp(-.5*(X-11.)**2 / 2.)
            yg1[X < 8.] = 0
            yg1[X > 14.1] = 0
            yg2 = 2*np.exp(-.5*(X-14.1)**2 / 1.)
            yg2[X < 14.1] = 0
            y = yg0 + yg1 + yg2
            y = 24.*y+60.
            
        elif self.meas_type == 'sim_Raman_real':
            X = X[:,None]
            y, y_var = self.func_prop.predict_f_compiled(X)
            y = y.numpy()
            
        return y

    def gamma_dist(self, x, a, beta):
        return beta**a * x**(a-1) * np.exp(-beta*x) / gamma(a)

    def measure_structure(self, X):
        # function for simulating measuring structure
        X = np.atleast_1d(X)
        if self.meas_type == 'sim':
            return self.struct_sim(X)
        elif self.meas_type == 'sim_Raman_real' or self.meas_type == 'sim_Raman_hard':
            return self.sim_Raman(X)
    
    def sim_Raman(self, X):
        X = np.atleast_1d(X)
        y = []
        for i in range(X.shape[0]):
            x = X[i]
            y_temp = self.gen_Raman(x)
            y.append(y_temp.flatten())
        y = np.asarray(y)
        return y
            
    def gen_Raman(self, x):
        m = self.struct_data['mn']
        u = self.struct_data['vr']
        if x <= 8.:
            y  = np.random.multivariate_normal(m[0,:].squeeze(), u[0,:,:].squeeze(), 1).T
        elif x > 14.1:
            y  = np.random.multivariate_normal(m[2,:].squeeze(), u[2,:,:].squeeze(), 1).T
        else:
            y  = np.random.multivariate_normal(m[1,:].squeeze(), u[1,:,:].squeeze(), 1).T
        yy = y + np.random.normal(0, np.sqrt(1E-6), y.shape)
        return yy[::30]
    
    def struct_sim(self, X):
        t = np.arange(0,1,.1)
        structure = np.zeros((X.shape[0], t.shape[0]))
        for idx, x in enumerate(X):
            if x <= .25:
                structure[idx,:] = 5.*np.exp(-1*((t-0.25)**2)/(1E-2)) + np.random.normal(0., .01, t.shape)
            else:
                structure[idx,:] = 5.*np.exp(-1*((t-0.75)**2)/(1E-2)) + np.random.normal(0., .01, t.shape)
        return np.round_(structure, 2)        
