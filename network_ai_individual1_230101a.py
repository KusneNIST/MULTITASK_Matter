# % A. Gilad Kusne, NIST, aaron.kusne@nist.gov, Release 3/1/2023
# % If using this work for a publication, please cite:
# % Kusne, A. Gilad, et al. "Scalable Multi-Agent Lab Framework for Lab Optimization" Matter 2023.

# % Packages used:
# % torch==1.11.0
# % tensorflow==2.8.2
# % tabulate==0.8.9
# % simpy==4.0.1
# % scipy==1.6.2
# % scikit-learn==0.24.1
# % pandas==1.2.4
# % numpy==1.20.1
# % matplotlib==3.3.4
# % gpflow==2.2.1

# % This software was developed by employees of the National Institute of
# % Standards and Technology (NIST), an agency of the Federal Government and
# % is being made available as a public service. Pursuant to title 17 United
# % States Code Section 105, works of NIST employees are not subject to
# % copyright protection in the United States.  This software may be subject
# % to foreign copyright.  Permission in the United States and in foreign
# % countries, to the extent that NIST may hold copyright, to use, copy,
# % modify, create derivative works, and distribute this software and its
# % documentation without fee is hereby granted on a non-exclusive basis,
# % provided that this notice and disclaimer of warranty appears in all
# % copies.

# % THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER
# % EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY
# % WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED
# % WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# % FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL
# % CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR
# % FREE.  IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT
# % NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES,
# % ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS
# % SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR
# % OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR
# % OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF
# % THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.


# Generate py file to create the agent class 'control_ai'.
# Abreviations:
# msir - material system internal representation
# spm - sample pool manager

from infrastructure_230101a import match_rows_in_matrix, normalize_each_row_by_sum, tern2cart, similarity_matrix
from global_variables_and_monitors_230101a import performance_monitor, mat_repository, agt_repository

import pandas as pd
import numpy as np
import math
import simpy
from tabulate import tabulate
from collections import namedtuple
pd.options.display.float_format = '{:,.2f}'.format

import torch
torch.set_default_dtype(torch.float64)
from torch.distributions import constraints
from torch.nn import Parameter, Softmax
from torch.nn.functional import one_hot

import pyro
from pyro.infer import MCMC, NUTS, HMC, Predictive
import pyro.contrib.gp as gp
import pyro.distributions as dist

from matplotlib import pyplot as plt
from matplotlib import cm
import tensorflow as tf
import gpflow
from gpflow.ci_utils import ci_niter
from gpflow.utilities import print_summary, set_trainable, to_default_float
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import fowlkes_mallows_score as fmi
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance as scipy_dist
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import spectral_embedding
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy, gmean
from IPython.display import display
from sklearn_extra.cluster import KMedoids

import tensorflow_probability as tfp
f64 = gpflow.utilities.to_default_float
from scipy.special import gamma
from scipy.spatial import Voronoi
from torch.nn.functional import one_hot



# example params
# agt_params = {'AL_method':'entropy', 'x_range':np.asarray([2., 17., 0.1]), 'phase_mapping':None, 'combined_acquisition':False, 'comb_acq_weight':0.,
#               'sample_pool_manager':spm_i,
#               'central_mat_repo':None, 'central_agt_repo':None,
#               'verbose':verbose}
# agt_params['phase_mapping'] = {'num_clusters':3, 'stdiv_XRD':.1, 'stdiv_C':1}

# example call:
# control_ai(env, 'funcprop', uuid4().int, self.synth, self.meas_fp, agt_params)

# Create AIs
class control_ai:
    def __init__(self, env, focus, unique_ID, synth, meas, params):
        self.msir = None # initialize the material system internal representation (MSIR) to None 
        self.spm = params['sample_pool_manager']
        self.meas = meas
        self.synth = synth
        self.env = env
        self.running_proc = env.process(self.ai_run()) # tie the AI to ai_run in the environment.
        self.focus = focus
        self.unique_ID = unique_ID
        self.cp_samples = []
        self.cp_max_std = -1
        self.curr_mean = []
        self.curr_var = []
        self.current_iteration_results = None
        self.perf_monitor = None
        self.BO_iter = 1
        self.verbose = params['verbose']
        self.save_figs = params['save_figs']
        self.phase_mapping_AL_method = params['AL_PM_method'] #'entropy'
        self.AL_BO_method = params['AL_BO_method']
        self.phase_mapping_params = params['phase_mapping'] #{'num_clusters':2, 'stdiv_XRD':.1, 'stdiv_C':1}
        self.changepoint = False
        self.x_range = params['x_range']
        self.combined_acquisition = False
        self.meas_type = params['meas_type']
        if params['use_cp']:
            self.use_cp = True
        if params['central_mat_repo'] is None:
            self.mat_repo = mat_repository(env, self.spm)
            self.use_coreg = False
            # print('set to repo object')
        else:
            self.mat_repo = params['central_mat_repo']
            self.agt_repo = params['central_agt_repo']
            self.use_coreg = True
            self.use_cp = True
            if params['combined_acquisition']:
                self.combined_acquisition = True
                self.comb_acq_weight = params['comb_acq_weight']
        
    def ai_run(self): 
        # The primary function for the AI.
        # Closed for loop for data collection, analysis, and decision making
        
        # if none of the samples in the sample pool have measurement, then measure two.
        yield self.env.process(self.get_first_data_if_needed()) 
        yield self.env.timeout(10) # This gives enough time for all the initial measurements to be taken.
        
        for i in range(10): # loops
            
            # update msir and print
            self.mat_repo.update_from_sample_pool()
            yield self.env.timeout(1)
            if self.verbose:
                print(f'{self.focus} {str(self.unique_ID)[:5]} starts analysis with msir: ')
                display(self.mat_repo.get_mat_data_all())
            
            # Run machine learning and active learning and store results
            xx = np.arange(self.x_range[0],self.x_range[1],self.x_range[2])[:,None] # prediction points
            xx = np.round_(xx,2)
            comp_sorted_by_acq = self.analysis_prediction_and_active_learning(xx)  
            yield self.env.timeout(1)
            
            # update history
            self.update_perf_monitor()
            force = False
            redo = True
            while redo:
                # collect the new data, synthesize new sample if needed.
                target_index, target_composition, measurement_result = yield self.env.process( self.collect_new_data_and_synth_if_needed(comp_sorted_by_acq, force) )
                yield self.env.timeout(2)

                # now that the sample is in msir, we can update its measurement value with the value measured.
                redo = yield self.env.process(self.mat_repo.update_with_measurement(self.focus, target_composition, measurement_result))

            yield self.env.timeout(1)
                
            # store results
            self.store_meas_data(target_composition, measurement_result)
            
            # plot
            plt.figure()
            self.plot_current_iteration_results(xx, target_composition)
       
    # Getting data -----------------------------------
    def get_first_data_if_needed(self):
        self.mat_repo.update_from_sample_pool() # update mat repo from sample pool
        repo = self.mat_repo.get_mat_data_all() # get the repo data
        idx = [i for i in range(repo.shape[0]) if repo[self.focus].iloc[i] is not None]
        if (len(idx) < 1):
            composition_ = np.asarray([repo['composition'].iloc[i] for i in range(repo.shape[0]) if (repo[self.focus].iloc[i] is None)])
            if self.verbose:
                print(f'{self.focus} {str(self.unique_ID)[:5]}, composition list: {composition_}')
            for i in range(2):
                if self.verbose:
                    print(f'{self.focus} {str(self.unique_ID)[:5]} getting data for c:{composition_[i]}')
                measurement_result = yield self.env.process(self.meas.measure(composition_[i]))
                redo = yield self.env.process(self.mat_repo.update_with_measurement(self.focus, composition_[i], measurement_result)) 
                if self.verbose:
                    print(f'{self.focus} {str(self.unique_ID)[:5]} data for c:{composition_[i]}, meas: {measurement_result}')
        yield self.env.timeout(1)
        
    def collect_new_data_and_synth_if_needed(self, comp_sorted_by_acq, force = False):  
        yield self.env.timeout(1)
        
        target_comp = comp_sorted_by_acq[0]
        # synthesize if needed.
        target_index = yield self.env.process( self.synth_if_needed(target_comp, force) )
               
        # measure
        measurement_result = yield self.env.process( self.collect_data_for_target_sample(target_comp, target_index) )
        
        return target_index, target_comp, measurement_result
    
    def collect_data_for_target_sample(self, target_composition, target_index):
        if type(target_composition) is np.ndarray:
            target_composition = target_composition.item()
        self.mat_repo.update_from_sample_pool() # update mat repo from sample pool
        # find which sample has target_composition
        with self.meas.sample_measure.request() as req:
            yield req # request measurement access
            print(f'{self.focus} {str(self.unique_ID)[:5]} requested measurement of c:{target_composition} at {self.env.now}')
            if self.verbose:
                print(spm.list_sample_pool_items())
            self.meas.target_composition = target_composition
            self.mat_repo.update_from_sample_pool() # update msir again
            # hand over control to the measurement instrument.
            measurement_result = yield self.env.process(self.meas.measure(target_composition)) # request measurement
            print(f'{self.focus} {str(self.unique_ID)[:5]} received measurement result for c:{target_composition} at {self.env.now}')
            self.mat_repo.update_from_sample_pool() # update msir

        return measurement_result
    
    def synth_if_needed(self, target_comp, force):
        if type(target_comp) is np.ndarray:
            target_comp = target_comp.item()
        self.mat_repo.update_from_sample_pool() # update msir again
        compositions_, sample_indices_ = self.mat_repo.get_compositions_and_sample_indices() # get the compositions of the samples in msir
        # match_index indicates which samples in msir match the target_composition
        match_index = match_rows_in_matrix(compositions_, target_comp)

        if not match_index.size or force: # if the target_composition is not in the pool
            with self.synth.sample_synthesis.request() as req: # synthesize the sample
                yield req # request synthesis access
                print(f'{self.focus} {str(self.unique_ID)[:5]} requested synthesis of c:{target_comp} at {self.env.now}')
                # synthesize sample
                self.synth.target_composition = target_comp
                # request that the sample is synthesized and wait until finished.
                # new sample is added to the sample_pool with the index new_sample_index
                new_sample_index = yield self.env.process(self.synth.synthesis(target_comp))
                print(f'{self.focus} {str(self.unique_ID)[:5]} synthesis complete of c:{target_comp} at {self.env.now}')
                # add new sample to internal representation
            self.mat_repo.update_from_sample_pool()
            target_index = new_sample_index # if synthesized, then the sample to measure is indicated by new_sample_index
        else:
            target_index = sample_indices_[match_index[0]] # if the sample is in the sample pool, then select that sample to be measured.
        return target_index
    
    # ML ----------------------------------------------
    def analysis_prediction_and_active_learning(self, X_full):
        data = self.get_training_data()
        if self.verbose:
            print(f'{self.focus}{str(self.unique_ID)[:5]} got training data')
        
        if self.use_coreg and self.use_cp:
            # print('clustering')
            cl, Ux = self.phase_mapping(data[2], data[3])
            # print('coreg:prediction') 
            X_FP=data[0]; Y_FP=data[1]; X_XRD=data[2]; Y_XRD=data[3]
    
            Fmu, Fvar, Pmu, Pvar, cp = self.bi_csp_full_Bayesian(X_full, X_XRD, cl, X_FP, Y_FP) # assumes num_clusters = 3
            self.cp_samples = cp
            self.curr_mean = Fmu.flatten()
            self.curr_var = Fvar.flatten()
            if self.cp_max_std < np.std(cp,axis=0).max():
                self.cp_max_std = np.std(cp,axis=0).max()
            if self.verbose:
                print('cp: clustered, prediction, AL,cp:' + self.focus[5:])
            if self.focus=='funcprop':
                training_data = (data[0], data[1])
                alpha = self.AL_BO(Fmu, Fvar, method=self.AL_BO_method, change_points = cp, Xpred = X_full)
                self.BO_iter = self.BO_iter + 1
                pred = Fmu; var_or_Cov = Fvar;
            elif self.focus=='structure':
                training_data = (data[2], cl)
                alpha = self.AL_PM(Pmu, Pvar, method=self.phase_mapping_AL_method, x = (data[2], X_full))
                pred = Pmu
            
                var_or_Cov = [] #np.sum(Pvar, axis=1).flatten();
            if self.combined_acquisition:
                # print('sizes to store in agt repo:',alpha.shape, pred.shape)
                indep_var = np.arange(self.x_range[0],self.x_range[1],self.x_range[2]).flatten()
                self.agt_repo.update_record(self.unique_ID, indep_var, alpha, pred)
    
        elif self.focus == 'funcprop': # if AI name is Mag, then perform GPR and GP-UCB Bayesian optimization
            # print('funcprop:starting GPR')
            training_data = (data[0], data[1])
            mean, var, Cov = self.func_prop_regression_gpflow_with_error_catching(data[0], data[1], X_full)
            # print('funcprop:AL')
            alpha = self.AL_BO(mean.flatten(), var.flatten(), method=self.AL_BO_method)
            self.BO_iter = self.BO_iter + 1
            pred = mean; var_or_Cov = Cov;
            if self.verbose:
                print('funcprop: GPR, AL')
            
        elif self.focus == 'structure': # if AI named xrd, perform GPC and active learning (variance maximization)
            # print('structure:phase mapping')
            cl, Ux = self.phase_mapping(data[0], data[1])
            training_data = (data[0], cl)
            
            if self.use_cp:
                U, U_var = self.bi_cs(data[0], cl, X_full)
            else: 
                # print('structure:prediction')
                U, U_var = self.phase_map_extrapolate(data[0], Ux, X_full)
                # print('structure:AL')
            alpha = self.AL_PM(U, U_var, method=self.phase_mapping_AL_method, x = (data[0], X_full))
            pred = U.copy()
            var_or_Cov = [] #np.sum(U_var, axis=1).flatten();
            if self.verbose:
                print('structure: phase map, prediction, AL')
            
        # selecting next sample
        query, comp_sorted_by_acq = self.pick_next_measurement(alpha, X_full)
        
        # store results
        self.store_ML_data(pred, var_or_Cov, alpha, comp_sorted_by_acq, training_data)
        
        return comp_sorted_by_acq
    
    def store_ML_data(self, pred, var_or_Cov, acquisition_function, composition_sorted_by_acq, training_data):
        if self.current_iteration_results is None:
            temp = {'prediction':[pred],'var_or_Cov':[var_or_Cov],'composition_sorted_by_acq':[composition_sorted_by_acq], \
                'acquisition_function':[acquisition_function],'training_data':[training_data]}
            self.current_iteration_results = temp
        else:
            self.current_iteration_results['prediction'].append(pred)
            self.current_iteration_results['var_or_Cov'].append(var_or_Cov)
            self.current_iteration_results['composition_sorted_by_acq'].append(composition_sorted_by_acq)
            self.current_iteration_results['acquisition_function'].append(acquisition_function)
            self.current_iteration_results['training_data'].append(training_data)
            
    def store_meas_data(self, x, y):
        if 'x' in self.current_iteration_results:
            x0 = self.current_iteration_results['x']
            y0 = self.current_iteration_results['y']
            self.current_iteration_results['x'] = np.concatenate((x0,x.flatten()[None,:]),axis=0)
            self.current_iteration_results['y'] = np.concatenate((y0,y.flatten()[None,:]),axis=0)
        else:
            self.current_iteration_results['x'] = x.flatten()[None,:]
            self.current_iteration_results['y'] = y.flatten()[None,:]
            
    def update_perf_monitor(self):
        yy = self.current_iteration_results['prediction'][-1]
        training_data = self.current_iteration_results['training_data'][-1]
        if self.focus == 'funcprop':
            if training_data[1].flatten().shape[0] == 2:
                entry = training_data[1].flatten()
            else:
                entry = np.atleast_1d( np.max(training_data[1]) )
            self.perf_monitor = performance_monitor.history_tracking(self.perf_monitor, entry)
        elif self.focus == 'structure':
            yy = np.argmax(yy, axis = 1)
            self.perf_monitor = performance_monitor.history_tracking(self.perf_monitor, yy.squeeze()[None,:])
    
    def get_training_data(self):
        if self.use_coreg:
            dataFP = self.mat_repo.get_mat_data('funcprop')
            dataXRD = self.mat_repo.get_mat_data('structure')
            data = (dataFP[0], dataFP[1], dataXRD[0], dataXRD[1])
        else:
            data = self.mat_repo.get_mat_data(self.focus)
        return data
    
    def get_training_data_one_source(self):
        data = self.mat_repo.get_mat_data(self.focus)
        return data

    def pick_next_measurement(self, alpha, X_full):
        # for the compositions sorted by aquisition values
        # remove those that are within 1E-2 of compositions in msir with measurement data.
        # This is done for the desired measurement type
        # print(f'pick next measurement, acq: {composition_sorted_by_acq[0:5].flatten()}')
        
        # !!!!!!!!!!!!! IF COMBINING ACQUISITION FUNCTIONS !!!!!!!!!!!!!!!!
        # Second condition here, only uses joint acq if the target is materials opt.
        if self.combined_acquisition and self.focus == 'funcprop':
            if self.verbose:
                print('inside combined acq')
                display(self.agt_repo.get_repo())
            # data_mean := {'indep_var': indep_var, 'acq_mean': acq_mean, 'pred_mean': 'Not used', \
            #          'mean_fp':mean_fp,'data_fp':data_fp,'mean_st':mean_st,'data_st':data_st}
            data_mean, data_all = self.agt_repo.get_other_records(self.unique_ID)
            stuff = [data_mean['mean_fp'], data_mean['data_fp'], data_mean['mean_st'], data_mean['data_st']]

            alpha_before = alpha.flatten().copy()           
            
            if data_mean['mean_st'] is not None:
                alpha_pm = normalize(data_mean['mean_st'].flatten())
                alpha_fp = normalize(alpha_before.flatten())
                
                cp_std = np.std(self.cp_samples , axis = 0)

                pm_weight = np.minimum(cp_std.max(),2.) / 2.
                fp_weight = 1 - pm_weight
                print('pm_weight,fp_weight', pm_weight, fp_weight)
                alpha = pm_weight*alpha_pm + fp_weight*alpha_fp
                
                if self.current_iteration_results is not None:
                    self.current_iteration_results['acquisition_function'].append(alpha.flatten())
            else:
                alpha = alpha_before      
            
            if data_mean['acq_mean'] is not None:
                
                print
                plt.figure(figsize = (12,2))
                plt.subplot(1,3,1)
                plt.plot(normalize(alpha_before))
                plt.title(f'{self.focus} {str(self.unique_ID)[:5]} alpha before')
                plt.subplot(1,3,2)
                plt.plot(normalize(data_mean['data_st']).T)
                plt.title('st data')
                plt.subplot(1,3,3)
                plt.plot(normalize(alpha))
                plt.title('alpha after')              
                
                if self.save_figs:
                    dn = r'G:\\My Drive\\Research\\jupyter\\Networked ML\\figs\\'
                    title_str = self.focus + str(self.unique_ID)[:5]
                    plt.savefig(dn + 'comb_acq_' + title_str + '_at_' + str(self.env.now) + '.svg', format='svg')
                    
                plt.show()

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
    
        comp_sorted_by_acq = X_full[np.argsort(-alpha.flatten())].flatten()
        comps2compare = np.empty((0))
        
        # composition for samples already measured.
        temp, _ = self.get_training_data_one_source()
        comps2compare = np.concatenate((comps2compare, temp.flatten()), axis = 0)
        
        # create list of samples currently being made
        synth_list = self.spm.synth_list.copy()
        if synth_list:
            temp = np.asarray(synth_list).flatten()
            comps2compare = np.concatenate((comps2compare, temp), axis = 0)
            
        # samples currently being measured
        lent_list = self.spm.lent_list.copy()
        if lent_list:
            comps_lent = np.asarray(lent_list).flatten()
            comps_lent_purpose = self.spm.lent_purpose.copy()
            if comps_lent.flatten().shape[0] != len(comps_lent_purpose):
                print('wtf: comps_lent:', comps_lent, 'purpose:', comps_lent_purpose)
            # If the sample is currently being measured for the same purpose, ignore.
            kp = [idx for idx, p in enumerate(comps_lent_purpose) if p == self.focus]
            if kp:              
                temp = comps_lent[kp]
                    
                # print('sizechallenge',lent_list, comps_lent, comps2compare.shape, temp.shape)
                comps2compare = np.concatenate((comps2compare, temp.flatten()), axis = 0)
    
        drop_idx = match_rows_in_matrix(comp_sorted_by_acq, comps2compare)
        new_comp_list = np.delete(comp_sorted_by_acq.flatten(), drop_idx, axis = 0)
        
        query = np.atleast_1d(new_comp_list[0])
        
        if self.verbose:
            print(f'orig:{comp_sorted_by_acq[0:5].flatten()}, new:{new_comp_list[0:5].flatten()}')
        return np.round_(query, decimals=2), np.round_(new_comp_list, decimals=2)

    # Phase Mapping -----------------------
    def similarity_matrix(self, Y, stdY):
        # Kernel used for spectral clustering,
        # print('sim',X.shape, Y.shape)
        d_Y = pairwise_distances(Y, metric = 'cosine')
        K_Y = np.exp(- d_Y**2 / stdY**2)
        return K_Y
    
    def phase_mapping(self, X, Y):
        stdX = self.phase_mapping_params['stdiv_C']
        stdY = self.phase_mapping_params['stdiv_XRD']
        num_clusters = self.phase_mapping_params['num_clusters']
        # various methods for performing phase mapping
        cl, Ux = self.phase_mapping_Kmedoids(X, Y, num_clusters)
        # cl, Ux = self.phase_mapping_ground_truth(X, num_clusters)
        return cl, Ux

    def phase_mapping_Kmedoids(self, X, Y, num_clusters):
        if X.shape[0] <= num_clusters:
            cluster_prob = np.eye(X.shape[0])
            cl = np.argmax(cluster_prob, axis=1).flatten()
        else:
            cl = KMedoids(n_clusters=num_clusters, metric = 'cosine', method = 'pam', random_state=0).fit(Y).labels_
            cl = cl.flatten()
            cluster_prob = self.hard_labels_to_Ux(cl,num_clusters)
        return cl, cluster_prob    
    
    def hard_labels_to_Ux(self,cl,N):
        Ux = np.zeros((cl.flatten().shape[0],N)).astype(int)
        idx = np.arange(0,cl.flatten().shape[0])
        Ux[idx,cl] = 1
        return Ux

    def phase_map_extrapolate(self, X, Ux, X_full):
        # extrapolate phase map results
        cl = np.argmax(Ux, axis = 1).flatten()
        U_full, U_var = self.phase_mapping_classification_with_error_catching(X, Ux, cl, X_full)
        return U_full, U_var
    
    def phase_mapping_classification_with_error_catching(self, X, Ux, cl, X_full):
        num_clusters = np.unique(cl).shape[0]
        N = X_full.shape[0]
        # If there is only 1 cluster, set the output values.
        if num_clusters == 1:
            mean = np.ones((N,1))
            var = np.zeros((N,1))
        else:
            counter = 0
        mean, var, m = self.phase_map_classification_gpflow(X, Ux, cl, X_full)
        return mean, var

    def phase_map_classification_gpflow(self, X, Ux, cl, X_full):
        cl = cl.flatten()[:,None]
        # Apply GPC to extrapolate phase region labels
        C = 3
        data = (X, cl) # create data variable that contains both the xy-coordinates of the currently measured samples and their labels.
        kernel = gpflow.kernels.Matern32() #+ gpflow.kernels.White(variance=0.01)   # sum kernel: Matern32 + White
        # Robustmax Multiclass Likelihood
        invlink = gpflow.likelihoods.RobustMax(C)  # Robustmax inverse link function
        likelihood = gpflow.likelihoods.MultiClass(C, invlink=invlink)  # Multiclass likelihood
        m = gpflow.models.VGP(data=data, kernel=kernel, likelihood=likelihood, num_latent_gps=C) # set up the GP model

        #m.likelihood.variance.assign(0.05)
        #p = m.likelihood.variance
        #m.likelihood.variance = gpflow.Parameter(p, transform=tfp.bijectors.Sigmoid(f64(0.01), f64(0.1)) )    

        opt = gpflow.optimizers.Scipy() # set up the hyperparameter optimization
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, method = 'tnc', options=dict(maxiter=ci_niter(1000)) ) # run the optimization
        f = m.predict_f(X_full) # what is the (non-squeezed) probabilistic function for extrapolating class labels over full XY coordinates
        y = m.predict_y(X_full) # what is the Poisson process for the full XY coordinates
        f_mean = f[0].numpy() # mean of f
        f_var = f[1].numpy() # variance of f
        y_mean = y[0].numpy() # mean of y
        y_var = y[1].numpy() # variance of y.
        return y_mean, y_var, m
    
    # ------- Regression -------------
    def func_prop_regression_gpflow(self, X, Y, X_full, minimize_method):
        #GPflow GPR for one scalar property.
        k = gpflow.kernels.SquaredExponential(lengthscales = [1])# + gpflow.kernels.White(variance=0.001) # set up kernel
        data = (tf.convert_to_tensor(X), tf.convert_to_tensor(Y.flatten()[:,None]))
        m = gpflow.models.GPR(data=data, kernel=k, mean_function=gpflow.mean_functions.Constant(Y.mean())) # set up GPR model

        m.likelihood.variance.assign(0.01)
        p = m.likelihood.variance
        m.likelihood.variance = gpflow.Parameter(p, transform=tfp.bijectors.Sigmoid(f64(0.001), f64(0.1)) )    

        opt = gpflow.optimizers.Scipy() # set up hyperparameter optimization
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, method = minimize_method, options=dict(maxiter=100))  # run optimization
        temp_mean, temp_var = m.predict_f(X_full) # compute the mean and variance for the other samples in the phase region
        _, temp_Cov = m.predict_f(X_full, full_cov = True)
        return temp_mean.numpy(), temp_var.numpy(), temp_Cov.numpy().squeeze()

    def func_prop_regression_gpflow_with_error_catching(self, X, Y, X_full):
        count = 0
        methods = ['tnc', 'BFGS', 'L-BFGS-B']
        minimize_method = methods[count]
        mean, var, Cov = self.func_prop_regression_gpflow(X, Y, X_full, minimize_method)
        return mean, var, Cov 
    
    def bi_csp_1D_full_Bayesian_analysis(self,Xpred, xs, ys, xf, yf, num_regions, numsteps, JIT = True):
        # Can use samples of change_point to define uncertainty.
        # can bin changepoint to make this faster and iterate over bins.
        data = [xs, ys, xf, yf, Xpred, num_regions]
        ker = NUTS(model_1D_joint_change_points_w_bounds_Bayesian, jit_compile=JIT, ignore_jit_warnings=True, max_tree_depth=3)
        posterior = MCMC(ker, num_samples=numsteps, warmup_steps=100)
        posterior.run(data);

        samples = {k: v.detach().cpu().numpy() for k, v in posterior.get_samples().items()}
        predictive = Predictive(model_1D_joint_change_points_w_bounds_Bayesian, posterior.get_samples())(data)

        return samples, predictive
    
    def bi_csp_full_Bayesian(self, Xpred, xs, ys, xf, yf):
        # Can use samples of change_point to define uncertainty.
        # can bin changepoint to make this faster and iterate over bins.
        num_regions = 3
        numsteps = 500
        
        data = [xs, ys, xf, yf, Xpred, num_regions]
        pyro.clear_param_store()
        samples, predictive = self.bi_csp_1D_full_Bayesian_analysis(Xpred, xs, ys, xf, yf, num_regions, numsteps, JIT = False)

        region_prob, fhat_mean, fhat_std, cp = self.bi_csp_1D_full_Bayesian_predict(Xpred, xf, yf, samples, predictive, num_regions, eps = 1E-6)

        # assumes only one functional property
        Fmu = fhat_mean.flatten()[:,None]
        Fvar = fhat_std.flatten()[:,None]
        Pmu = region_prob
        Pvar = [] 
        
        return Fmu, Fvar, Pmu, Pvar, cp
    
    def bi_csp_1D_full_Bayesian_predict(self, Xpred, xf, yf, samples, predictive, num_regions, eps = 1E-6):
        Nnew = Xpred.shape[0]
        Nsamples_GPR = 10
        eps = 1E-6
        cp_samples = samples['changepoint_'] 

        if num_regions == 1:
            cp_ =cp_samples.squeeze(-1)
        else:
            cp_ = np.sort(cp_samples,axis=1)
        cp = cp_.mean(axis=0)
        region_prob = cp_samples_to_class_prob(cp_, Xpred)

        # !!!!! WE SAMPLE FROM MVN using mean and COV and output the sample
        # ! 1 sample gives good approx to results when using 10 samples.
        fhat_Xpred = np.zeros((samples['changepoint_'].shape[0]*Nsamples_GPR,Xpred.shape[0]))
        for i in range(samples['changepoint_'].shape[0]):
            mean, cov, _, iseg = gpr_piecewise_forward_w_mean(Xpred, xf, yf, samples['changepoint_'][i,:], \
                                                        samples['gp_var'][i,:], samples['gp_lengthscale'][i,:], \
                                                        samples['gp_noise'][i], predictive['prior_means'][i])
            mean_Xpred = torch.cat(mean)
            # print('mean_Xpred:', mean_Xpred.flatten())
            cov_Xpred = torch.zeros((Xpred.shape[0], Xpred.shape[0]))
            for j in range(len(iseg)):
                seg_curr = np.argwhere(iseg[j]).flatten()
                cov_Xpred[seg_curr[:,None],seg_curr] = to_torch(cov[j])
            temp = cov_Xpred + torch.eye(Nnew) * eps
            idx = torch.arange(Nsamples_GPR*i,Nsamples_GPR*(i+1))
            fhat_Xpred[idx,:] = dist.MultivariateNormal(mean_Xpred.flatten(), cov_Xpred + torch.eye(Nnew) * eps).sample((Nsamples_GPR,))

        fhat_mean = fhat_Xpred.mean(axis=0)
        fhat_std = fhat_Xpred.std(axis=0)

        return region_prob, fhat_mean, fhat_std, cp_

    def bi_cs(self, xs, ys, X_pred):
        # Can use samples of change_point to define uncertainty.
        # can bin changepoint to make this faster and iterate over bins.
        data = [xs, ys, X_pred]

        ker = NUTS(model_pm_1D_CP_w_bounds, jit_compile=False, ignore_jit_warnings=True, max_tree_depth=3)
        posterior = MCMC(ker, num_samples=100, warmup_steps=10)
        posterior.run(data);
        
        s = {k: v.detach().cpu().numpy() for k, v in posterior.get_samples().items()}
        cp_ = np.sort(s['changepoint_'],axis=1)

        Pmu = cp_samples_to_class_prob(cp_, X_pred)
        Pvar = []
        
        return Pmu, Pvar
    
    # Active Learning ----------------
    def AL_BO(self, mean, var, method='UCB', change_points = None, Xpred = None):
        if method=='UCB':
            Dsize = mean.shape[0]
            alpha = self.gp_ucb(Dsize, mean, var, BO_lambda = .1)
        elif method=='UCB+CP':
            Dsize = mean.shape[0]
            alpha = self.gp_ucb_cp(Dsize, mean, var, change_points, Xpred, BO_lambda = .1)            
        return alpha

    def gp_ucb(self, Dsize, mean, var, BO_lambda = .1):
        # compute utility of samples using 'gaussian process - upper confidence bound' and then select sample with highest utility
        BO_beta = 2*np.log(Dsize * (self.BO_iter**2) * (np.pi**2) / (6 * BO_lambda) ) # GP-UCB
        BO_alpha = mean + np.sqrt(BO_beta) * np.sqrt(var)
        self.BO_iter = self.BO_iter + 1
        return BO_alpha
    
    def gp_ucb_cp(self, Dsize, mean, var, cp, Xpred, BO_lambda = .1):
        # compute utility of samples using 'gaussian process - upper confidence bound' and then select sample with highest utility
        d = pairwise_distances(Xpred.flatten()[:,None], cp.flatten()[:,None])
        d = np.min(d, axis = 1)
        r = np.exp(-.5*d**2 / 1).flatten()
        if len(var.shape) == 2:
            r = r.flatten()[:,None]
        
        BO_beta = 2*np.log(Dsize * (self.BO_iter**2) * (np.pi**2) / (6 * BO_lambda) ) # GP-UCB
        BO_alpha = mean + np.sqrt(BO_beta) * np.sqrt(var)
        self.BO_iter = self.BO_iter + 1
        # print('UCB:', BO_alpha.shape, BO_alpha.flatten())
        # print('r:',r.shape, r.flatten())
        BO_alpha = BO_alpha + r
        # print('UCB+r:', BO_alpha.shape, BO_alpha.flatten())
        
        return BO_alpha

    def AL_PM(self, U, U_var, method='entropy', x=None):
        # update process bar
        # active learning schemes applied to phase mapping objective.
        if method == 'entropy':
            alpha = entropy(U, axis=1).flatten()
        return alpha 
        
    # Plot ----------------------------------
    
    # plot GP regression results
    def plot_gpr(self, X, m, C, training_points=None, acq=None, query=None, history=None):
        X, m, C = to_numpy([X, m, C])
        if len(C.squeeze().shape) == 2:
            C = np.diag(C)
        # print('inside gpr plot',X.shape,m.shape,C.shape)
        plt.figure(figsize = (12,2))
        plt.subplot(1,3,1)
        plt.fill_between(X.flatten(), m.flatten() - 1.96*np.sqrt(C.flatten()), m.flatten() + 1.96*np.sqrt(C.flatten()), alpha=0.5);
        plt.plot(X, m, "-");
        title_str = self.focus + str(self.unique_ID)[:5] 
        if self.use_coreg:
            title_str = title_str + ' coreg'
        if self.combined_acquisition:
            title_str = title_str + ' comb_acq'
        plt.title(title_str + ' at ' + str(self.env.now))
        if training_points is not None:
            X_, Y_ = training_points
            plt.plot(X_, Y_, "kx", mew=2);
        if acq is not None:
            plt.subplot(1,3,2)
            plt.plot(X, acq)
        if query is not None:
            minTS = np.min(acq) 
            maxTS = np.max(acq) 
            plt.subplot(1,3,1)
            plt.plot([query, query], [minTS, maxTS], 'm');
        if history is not None:
            plt.subplot(1,3,3)
            simple_regret = performance_monitor.simple_regret(history, self.meas_type)
            plt.plot(simple_regret);
            # plt.ylabel('%')

        if self.save_figs:
            dn = r'G:\\My Drive\\Research\\jupyter\\Networked ML\\figs\\'
            plt.savefig(dn + 'gpr_' + title_str + '_at_' + str(self.env.now) + '.svg', format='svg')
    
        plt.show();
        
    # plot GP classification results
    def plot_gpc(self, X, label_curves, var_or_Cov, training_points=None, acq=None, query=None, history=None):
        plt.figure(figsize = (12,2))
        plt.subplot(1,3,1)
        plt.plot(X, label_curves)
        #plt.ylim([-.1, 1.1])
        plt.plot([query, query],[0,1],'m');
        title_str = self.focus + str(self.unique_ID)[:5]
        if self.use_coreg:
            title_str = title_str + ' coreg'
        if self.combined_acquisition:
            title_str = title_str + ' comb_acq'
        plt.title(title_str + ' at ' + str(self.env.now))
        
        if training_points is not None:
            X_, Y_ = training_points
            plt.plot(X_,Y_/2, 'kx', mew=2);
            
        plt.subplot(1,3,2)
        plt.plot(X, acq)
        plt.title('acq')
        if history is not None:
            plt.subplot(1,3,3)
            fmi = performance_monitor.phase_mapping_performance(history, self.x_range)
            plt.plot(fmi);
        
        if self.save_figs:
            dn = r'G:\\My Drive\\Research\\jupyter\\Networked ML\\figs\\'
            plt.savefig(dn + 'gpc_' + title_str + '_at_' + str(self.env.now) + '.svg', format='svg')

        plt.show();
        
    def plot_current_iteration_results(self, xx, query):
        yy = self.current_iteration_results['prediction'][-1]
        var_or_Cov = self.current_iteration_results['var_or_Cov'][-1]
        training_data = self.current_iteration_results['training_data'][-1]
        acq = self.current_iteration_results['acquisition_function'][-1]
        # print(xx.shape, yy.shape, var_or_Cov.shape, acq.shape)
        if self.focus == 'funcprop':
            self.plot_gpr(xx, yy, var_or_Cov, training_points=training_data, acq = acq, query = query, history = self.perf_monitor)
        elif self.focus == 'structure':
            self.plot_gpc(xx, yy, var_or_Cov, training_points=training_data, acq = acq, query = query, history = self.perf_monitor)

        
# ------------ Support Functions ------------------------

# Support -----------------------------
def to_numpy(v):
    for i in range(len(v)):
        if type(v[i]) is not np.ndarray:
            v[i] = v[i].numpy()
    return v

def to_torch(v):
    if not torch.is_tensor(v):
        v = torch.tensor(v)
    return v

def make_2d(v):
    for i in range(len(v)):
        if len(v[i].shape == 1):
            v[i] = v[i][:,None]
    return v
    
def change_points_to_labels(cp, X):
    if type(X) is not np.ndarray:
        X = X.numpy()
    cp = np.sort(cp)
    cl = np.zeros((X.shape[0])).astype(np.compat.long)
    N = cp.shape[0] # N = 3
    for i in np.arange(0, N):
        if i < N-1:
            idx = np.logical_and(X > cp[i], X < cp[i+1])
        elif i == N-1:
            idx = X > cp[i]
        cl[idx.flatten()] = i+1
    return cl

def change_points_to_labels_torch(cp, X):
    if type(X) is np.ndarray:
        X = torch.tensor(X)
    cp,_ = torch.sort(cp)
    cl = torch.zeros((X.shape[0])).long()
    N = cp.shape[0] # N = 3
    for i in range(0, N):
        if i < N-1:
            idx = torch.logical_and(X > cp[i], X < cp[i+1])
        elif i == N-1:
            idx = X > cp[i]
        cl[idx.flatten()] = i+1
    return cl
    
def one_hot_np(v, cp, X):
    oh = np.zeros((v.shape[0], cp.shape[0] + 1))
    oh[np.arange(v.shape[0]),v] = 1
    return oh

def cp_samples_to_class_prob(cp_, X):
    # print(cp_.shape, X.shape)
    if type(X) is not np.ndarray:
        X = X.numpy()
    # cl_ = np.zeros((X.shape[0],cp_.shape[0]))
    one_hot_samples = np.zeros((cp_.shape[0],X.shape[0],cp_.shape[1]+1))
    # print(one_hot_samples.shape)
    for i in range(cp_.shape[0]):
        cl = change_points_to_labels(cp_[i,:], X)
        # print(cp_[i,:], np.unique(cl))
        one_hot_samples[i,:,:] = one_hot_np(cl, cp_[i,:], X)
    probs = np.mean(one_hot_samples, axis = 0)
    return probs

def model_1D_joint_change_points_w_bounds_Bayesian(data):
    # print('starting')
    noise=torch.tensor(0.01)
    jitter=torch.tensor(1.0e-5)
    
    xs = to_torch(data[0])
    ys = to_torch(data[1])
    xf = to_torch(data[2])
    yf = to_torch(data[3])
    Xpred = to_torch(data[4])
    num_regions = to_torch(data[5])
 
    if len(yf.shape) == 1:
        yf = yf[:,None]
        
    uL, _ = torch.sort(ys)
    if uL.shape[0] == 2:
        num_regions = 2
        for i in range(2):
            ys[ys == uL[i]] = i
    
    Xsf = torch.vstack((xs,xf))
    idx_st = torch.arange(xs.shape[0])
    idx_fp = torch.arange(xf.shape[0]) + xs.shape[0]
    # print('before sampling')
    
    changepoint_bounds_min, changepoint_bounds_max = bounds_set_csp(num_regions, xs, ys)
    # print('bounds', changepoint_bounds_min, changepoint_bounds_max)
    gp_var_bound_min = 1.*torch.ones((num_regions,yf.shape[1])).double()
    gp_var_bound_max = 20.*torch.ones((num_regions,yf.shape[1])).double()
    gp_lengthscale_bound_min = 1.*torch.ones((num_regions,yf.shape[1])).double()
    gp_lengthscale_bound_max = 20.*torch.ones((num_regions,yf.shape[1])).double()
    
    changepoint_ = pyro.sample('changepoint_', dist.Uniform(changepoint_bounds_min.flatten(),changepoint_bounds_max.flatten())).double()
    gp_noise = pyro.sample("gp_noise", dist.Uniform(0.01, .1)).double()
    gp_var = pyro.sample("gp_var", dist.Uniform(gp_var_bound_min, gp_var_bound_max)).double()
    gp_lengthscale = pyro.sample("gp_lengthscale", dist.Uniform(gp_lengthscale_bound_min, gp_lengthscale_bound_max)).double()
    
    cluster_labels = change_points_to_labels_torch(changepoint_, Xsf)
    membership = one_hot(cluster_labels.long()).double()
    
    region_labels = change_points_to_labels_torch(changepoint_, Xsf)
    membership = one_hot(region_labels, num_regions)

    mean_full = torch.zeros((xf.shape[0],yf.shape[1])).double()
    kernel_full = torch.zeros((xf.shape[0],xf.shape[0],yf.shape[1])).double()
    prior_means = yf.flatten().mean()*torch.ones((num_regions, yf.shape[1])).double() # THIS SHOULD BE UPDATED!!!
    
    yf_mean_removed = torch.clone(yf).double()
    
    for i in range(num_regions):
        idx = torch.argwhere(region_labels[idx_fp] == i).flatten() # just fp data points

        if idx.numel() > 0:
            for j in range(yf.shape[1]):
                xx = xf[idx,:].double()
                yy = yf[idx,j][:,None].double() - yf[idx,j].mean().double()
                
                prior_means[i,j] = yf[idx,j].mean().double()

                mean_xx_post, K_xx_post = gp_forward(xx, xx, yy, gp_var[i,j], gp_lengthscale[i,j], gp_noise)
                mean_full[idx,j]=mean_xx_post.flatten().double()
                kernel_full[idx.flatten()[:,None],idx.flatten(),j]=K_xx_post.double()

                yf_mean_removed[idx,j] = yf[idx,j].double() - yf[idx,j].mean().double()
    
    pyro.deterministic('prior_means', prior_means)
    # print('kernel_full:', kernel_full.squeeze())
    
    for j in range(yf.shape[1]):
        kernel_full[:,:,j] += jitter * torch.eye(xf.shape[0]).double()
    
    # print(changepoint_, mean_full.flatten(), kernel_full.squeeze(), yf_mean_removed.flatten())
    pyro.sample("obs", dist.MultivariateNormal(loc=mean_full.flatten(), covariance_matrix=kernel_full.squeeze()), obs=yf_mean_removed.flatten())

def gpr_piecewise_forward_w_mean(Xpred, Xtrain, Ytrain, cp, var, lengthscale, noise, prior_for_means):
    # print('prior_means:', prior_means)
    Xtrain = to_torch(Xtrain)
    Ytrain = to_torch(Ytrain)
    Xpred = to_torch(Xpred)
    cp = to_torch(cp)
    var = to_torch(var)
    lengthscale = to_torch(lengthscale)
    noise = to_torch(noise)
    prior_for_means = to_torch(prior_for_means.flatten())
    
    cp, _ = torch.sort(to_torch(cp.flatten()))
    cp = cp.flatten()
    
    # predict segment before first changepoint.
    idx_pred = Xpred[:,0] < cp[0]
    idx_train = Xtrain[:,0] < cp[0]
    m0, c0 = gp_forward(Xpred[idx_pred,:], Xtrain[idx_train,:], Ytrain[idx_train,:]-prior_for_means[0], var[0], lengthscale[0], noise)
    mean = [m0+prior_for_means[0]]
    cov = [c0]
    xsegment = [Xpred[idx_pred,:]]
    idxsegment = [idx_pred]
    
    # predict all other segments.
    for i in range(cp.shape[0]):
        if i == cp.shape[0]-1:
            idx_pred = Xpred[:,0] > cp[i]
            idx_train = Xtrain[:,0] > cp[i]
        else:
            idx_pred = torch.logical_and(Xpred[:,0] > cp[i], Xpred[:,0] < cp[i+1])
            idx_train = torch.logical_and(Xtrain[:,0] > cp[i], Xtrain[:,0] < cp[i+1])
            
        m_curr, c_curr = gp_forward(Xpred[idx_pred,:], Xtrain[idx_train,:], Ytrain[idx_train,:]-prior_for_means[i+1], var[i+1], lengthscale[i+1], noise)
        mean.append(m_curr+prior_for_means[i+1])
        cov.append(c_curr)
        xsegment.append(Xpred[idx_pred,:])
        idxsegment.append(idx_pred)
    
    return mean, cov, xsegment, idxsegment

def gp_forward(Xtest, Xtrain, Ytrain, var, lengthscale, noise):
    # Derived from: https://num.pyro.ai/en/0.7.1/examples/gp.html
    k_pp = gpkernel(Xtest, Xtest, var, lengthscale, noise, include_noise=True)
    k_pX = gpkernel(Xtest, Xtrain, var, lengthscale, noise, include_noise=False)
    k_XX = gpkernel(Xtrain, Xtrain, var, lengthscale, noise, include_noise=True)

    #k_XX = torch.tensor( nearestPD(k_XX.detach().numpy()) ).double()
    K_xx_inv = torch.linalg.inv(k_XX.double()).double()
    # print(k_XX.shape, K_xx_inv.shape, Xtest.shape, Xtrain.shape, Ytrain.shape)
    K_xx_post = k_pp.double() - torch.matmul(k_pX.double(), torch.matmul(K_xx_inv.double(), k_pX.T.double())).double()
    mean_xx_post = torch.matmul(k_pX.double(), torch.matmul(K_xx_inv.double(), Ytrain.double()))
    return mean_xx_post, K_xx_post

def gpkernel(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
    deltaXsq = torch.pow((X.double() - Z.T.double()) / length.double(), 2.0)
    k = var.double() * torch.exp(-0.5 * deltaXsq).double()
    if include_noise:
        k += (noise.double() + jitter) * torch.eye(X.shape[0]).double()
    return k

def bounds_set_csp(num_regions, xs, ys):
    xs = xs.flatten()
    ys = ys.flatten()
    bounds_min = torch.ones((num_regions-1,1))*xs.min()
    bounds_max = torch.ones((num_regions-1,1))*xs.max()
    uL = torch.unique(ys)
    if uL.shape[0] == 2:
        bounds_min = torch.ones((num_regions-1,1))*xs[ys == 0].max()
        idx_y_is_1_or_2 = torch.logical_or(ys == 1, ys == 2)
        bounds_max = torch.ones((num_regions-1,1))*xs[idx_y_is_1_or_2].min()
    if uL.shape[0] > 2:
        bounds_points = torch.zeros((uL.shape[0],2))
        bounds_min = []
        bounds_max = []
        for i in range(uL.shape[0]):
            bounds_points[i,:] = torch.tensor([xs[ys == uL[i]].min(), xs[ys == uL[i]].max()])
        sorted_bounds, _ = torch.sort(bounds_points.flatten())
        
        for i in range(1, sorted_bounds.shape[0]-2, 2):
            bounds_min.append(sorted_bounds[i])
            bounds_max.append(sorted_bounds[i+1])

    bounds_min = torch.tensor(bounds_min)
    bounds_max = torch.tensor(bounds_max)
    return bounds_min, bounds_max

def change_points_to_labels_torch(cp, X):
    if type(X) is np.ndarray:
        X = torch.tensor(X)
    cp,_ = torch.sort(cp)
    cl = torch.zeros((X.shape[0])).long()
    N = cp.shape[0] # N = 3
    for i in range(0, N):
        if i < N-1:
            idx = torch.logical_and(X > cp[i], X < cp[i+1])
        elif i == N-1:
            idx = X > cp[i]
        cl[idx.flatten()] = i+1
    return cl

def model_pm_1D_CP_w_bounds(data):
    num_regions = 3
    noise=torch.tensor(0.01)
    jitter=torch.tensor(1.0e-5)
    
    xs = to_torch(data[0])
    ys = to_torch(data[1])
        
    uL, _ = torch.sort(ys)
    if uL.shape[0] == 2:
        num_regions = 2
        for i in range(2):
            ys[ys == uL[i]] = i
    
    changepoint_bounds_min, changepoint_bounds_max = bounds_set_csp(num_regions, xs, ys)
    changepoint_ = pyro.sample('changepoint_', dist.Uniform(changepoint_bounds_min.flatten(),changepoint_bounds_max.flatten())).double()

    region_labels = change_points_to_labels_torch(changepoint_, xs)
    membership = one_hot(region_labels, num_regions)

    # print('look here:', xs.flatten(), ys.flatten(), changepoint_.flatten())
    pyro.sample('obs', dist.Categorical(logits=membership), obs=ys.flatten().double())   

def normalize(v):
    v = np.ma.array(v, mask=np.isnan(v))
    if len(v.shape) == 1:
        nv = (v - v.min())/(v.max()-v.min())
    else:
        nv = v.copy()
        for i in range(v.shape[0]):
            nv[i,:] = (v[i,:] - v[i,:].min())/(v[i,:].max()-v[i,:].min())
    return nv

def cosd(deg):
    # cosine with argument in degrees
    return np.cos(deg * np.pi/180)

def sind(deg):
    # sine with argument in degrees
    return np.sin(deg * np.pi/180)

def tern2cart(T):
    # convert ternary data to cartesian coordinates
    sT = np.sum(T,axis = 1)
    T = 100 * T / np.tile(sT[:,None],(1,3))

    C = np.zeros((T.shape[0],2))
    C[:,1] = T[:,1]*sind(60)/100
    C[:,0] = T[:,0]/100 + C[:,1]*sind(30)/sind(60)
    return C

# Data Storage -------------------------
def set_dynamic_params(reset_storage_variables = False, reset_rep_variables = False, store_results = False):
    # setting up variables for storing data
    global params_dynamic
    if reset_rep_variables:
        params_dynamic['BO_iter'] = 1
        params_dynamic['iter'] = 1
        params_dynamic['phase_map_converged'] = False
        params_dynamic['first_iter'] = True    
    
    if reset_storage_variables:
        params_dynamic['cluster_results_by_rep'] = []
        params_dynamic['measured_samples_XRD'] = []
        params_dynamic['measured_samples_FP'] = []
        params_dynamic['measured_samples_XRD_by_rep'] = []
        params_dynamic['measured_samples_FP_by_rep'] = []
        params_dynamic['measured_samples_XRD'] = []
        params_dynamic['measured_samples_FP'] = []
        params_dynamic.pop('cluster_results', None) # remove this field. use its absence to populate in 'store' function.

    # this part happens at the end of a CAMEO rep.
    if store_results:
        params_dynamic['cluster_results_by_rep'].append(params_dynamic['cluster_results'])
        params_dynamic.pop('cluster_results', None) # remove this field. use its absence to populate in 'store' function.

        if params['method'][:5] == 'coreg':
            params_dynamic['measured_samples_XRD_by_rep'].append(params_dynamic['measured_samples_XRD'])
            params_dynamic['measured_samples_FP_by_rep'].append(params_dynamic['measured_samples_FP'])
