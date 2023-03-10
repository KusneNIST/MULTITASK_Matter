
from infrastructure_230101a import match_rows_in_matrix

import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
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
from IPython.display import display

class sample_pool_manager:
    def __init__(self, env, name):
        self.name = name
        self.env = env
        self.sample_index_max = 0
        self.sample_pool = simpy.FilterStore(env)
        self.lent_list = []
        self.lent_purpose = []
        self.synth_list = []
        self.status = []
        self.all_synthesized = [2., 17.]
    def get_sample_index_max(self):
        return self.sample_index_max
    def increment_sample_index_max(self):
        self.sample_index_max = self.sample_index_max + 1
        return self.sample_index_max
    def list_sample_pool_items(self):
        return self.sample_pool.items.copy()
    def lent_list_add(self, comp, meas_method):
        # print('lent sample indices', lent_sample_indices)
        self.lent_list.append(comp)
        self.lent_purpose.append(meas_method)   
    def lent_list_remove(self, comp):
        # print('inside put', sample_index, self.lent_list)
        idx = np.where(self.lent_list == comp)[0]
        if idx.flatten().shape[0] > 0:
            del(self.lent_list[idx[0].item()])
            del(self.lent_purpose[idx[0].item()])
    def sample_pool_append(self, items):
        if isinstance(items, list):
            for item in items:
                self.sample_pool.items.append(item)
                self.sample_index_max = self.sample_index_max + 1
        else:
            self.sample_pool.items.append(items)
            self.sample_index_max = self.sample_index_max + 1
    def synth_list_add(self,comp):
        if type(comp) is np.ndarray:
            comp = comp.item()
        self.synth_list.append(comp)
    def synth_list_remove(self,comp):
        drop_idx = [idx for idx,c in enumerate(self.synth_list) if c == comp]
        if drop_idx:
            del(self.synth_list[drop_idx[0]])
    def get_sample(self, composition):
        condition = lambda sample: sample.composition == composition
        cont_wait = True
        while cont_wait:
            items = self.list_sample_pool_items()
            compositions = np.asarray([item.composition for item in items])
            # print('compositions in spm:', compositions.flatten())
            if composition in compositions:
                sample = yield self.sample_pool.get(condition)
                cont_wait = False
            else:
                print(f'waiting for c:{composition}')
                yield self.env.timeout(1)
        return sample   
    def put_sample(self, sample):
        self.sample_pool.items.append(sample)
        yield self.env.timeout(1)

class performance_monitor:
    # class for monitoring active learning phase mapping and bayesian materials optimization performance
    def history_tracking(history, new_record):
        if history is None:
            history = new_record
        else:
            # print(f'history concat {history.shape}, {new_record.shape}')
            history = np.concatenate((history,new_record),axis=0)
        return history
    def phase_mapping_performance(phase_map_estimate, x_range):
        # phase mapping performance using FMI measure
        # assume phase_map_estimate is a matrix of row vectors
        xx = np.arange(x_range[0],x_range[1],x_range[2]).flatten()[None,:]
        LABELS_TRUE = np.zeros(xx.shape)
        LABELS_TRUE[xx > 8] = 1
        LABELS_TRUE[xx > 14.1] = 2
        measure_fmi = np.zeros((phase_map_estimate.shape[0],1))
        for i in range(phase_map_estimate.shape[0]):
            # print('inside fmi for loop:', LABELS_TRUE.shape, phase_map_estimate[i,:].shape )
            measure_fmi[i] = fmi(LABELS_TRUE.squeeze(), phase_map_estimate[i,:].squeeze())
        return measure_fmi
    def simple_regret(val_by_iter, meas_type):
        # simple regret performance measure
        if meas_type == 'sim_Raman_real':
            TRUE_MAX = 109.92
        elif meas_type == 'sim_Raman_hard':
            TRUE_MAX = 110.23
        measure_regret = np.zeros(val_by_iter.shape)
        max_by_iter = np.maximum.accumulate(val_by_iter)
        simple_regret = TRUE_MAX - max_by_iter
        return simple_regret/TRUE_MAX

class agt_repository:
    def __init__(self):
        self.agt_repo = None   
    def init_agt_repo(self, AI_list):
        ID_list = np.asarray( [entry.unique_ID for entry in AI_list] ).squeeze()
        AI_type = [entry.focus for entry in AI_list]
        agt_focus_list = [entry.focus for entry in AI_list]
        n_ = [None for i in range(ID_list.shape[0])]
        data = {'unique_ID': ID_list, 'AI_type': AI_type, 'indep_var': n_, 'curr_acq': n_, 'curr_pred': n_}
        data_repository = pd.DataFrame(data = data) # hand off info to the central representation.
        self.agt_repo = data_repository
    def update_record(self, unique_ID, indep_var, curr_acq, curr_pred):
        # Assumes inputs are vectors and converts to single entry list.
        # print('update_record: curr_acq:', curr_acq[:5])
        # print('update_record: norm curr_acq:', normalize(curr_acq.flatten())[:5])
        df_index_to_update = self.agt_repo[self.agt_repo['unique_ID']==unique_ID].index[0] # find the sample to update with measurements.
        df_index_to_update = np.atleast_1d(df_index_to_update)
        self.agt_repo['indep_var'].iloc[df_index_to_update] = [indep_var] # update value
        self.agt_repo['curr_acq'].iloc[df_index_to_update] = [normalize(curr_acq.flatten())] # update value
        self.agt_repo['curr_pred'].iloc[df_index_to_update] = [curr_pred] # update value
    def get_other_records(self, unique_ID):
        df_indices_to_grab = self.agt_repo[self.agt_repo['unique_ID'] != unique_ID].index.to_numpy()
        # print('indices:', df_indices_to_grab)
        df_indices_to_grab = np.atleast_1d(df_indices_to_grab)
        indep_var = self.agt_repo['indep_var'].iloc[df_indices_to_grab[0]]
        acq_mean, acq_all = self.series_mean('curr_acq', df_indices_to_grab)
        mean_fp, data_fp, mean_st, data_st = self.type_mean('curr_acq', df_indices_to_grab)
        # pred_mean, pred_all = self.series_mean('curr_pred', df_indices_to_grab)
        data_mean = {'indep_var': indep_var, 'acq_mean': acq_mean, 'pred_mean': 'Not used', \
                     'mean_fp':mean_fp,'data_fp':data_fp,'mean_st':mean_st,'data_st':data_st}
        data_all = {'indep_var': indep_var, 'acq_all': acq_all, 'pred_all': 'Not used'}
        return data_mean, data_all
    def get_repo(self):
        return self.agt_repo
    def series_mean(self, col, idx):
        temp = self.agt_repo[col].iloc[idx].to_numpy()
        data = [entry.flatten() for entry in temp if entry is not None]
        series_mean = None
        if data:
            data = np.vstack( data )
            data = np.ma.array(data, mask=np.isnan(data))
            series_mean = data.mean(axis = 0).flatten()
        return series_mean, data
    def type_mean(self, col, idx):
        temp = self.agt_repo[col].iloc[idx].to_numpy()
        AI_type = self.agt_repo['AI_type'].iloc[idx].to_numpy() # 'funcprop', 'structure'
        
        data_fp = [entry.flatten() for count, entry in enumerate(temp) if entry is not None and AI_type[count]=='funcprop']
        data_st = [entry.flatten() for count, entry in enumerate(temp) if entry is not None and AI_type[count]=='structure']
        
        mean_fp = None
        mean_st = None
        if data_fp:
            data_fp = np.vstack( data_fp )
            data_fp = np.ma.array(data_fp, mask=np.isnan(data_fp))
            mean_fp = data_fp.mean(axis = 0).flatten()
        if data_st:
            data_st = np.vstack( data_st )
            data_st = np.ma.array(data_st, mask=np.isnan(data_st))
            mean_st = data_st.mean(axis = 0).flatten()
        return mean_fp, data_fp, mean_st, data_st
    
class mat_repository:
    def __init__(self, env, spm):
        self.spm = spm
        self.env = env
        self.mat_repo = self.init_mat_repo()
    def init_mat_repo(self):
        # get the compositions in the sample pool.
        items = self.spm.list_sample_pool_items()
        compositions = np.asarray([s.composition for s in items])
        compositions = np.stack(compositions, axis = 0)
        # initialize msir entries for each sample in sample pool.
        n_ = [None for i in range(compositions.shape[0])]
        data = {'sample_index': np.arange(compositions.shape[0]),'composition': compositions, 'structure': n_, 'funcprop': n_}
        mat_repo = pd.DataFrame(data = data) # hand off info to the central representation.
        # display(mat_repo)
        return mat_repo        
    def get_mat_data_all(self):
        db = self.mat_repo
        return db
    def get_mat_data(self, data_type):
        data = self._get_mat_data(data_type)
        return data
    def get_compositions(self):
        return self.mat_repo['composition'].to_numpy()
    def get_compositions_and_sample_indices(self):
        return self.mat_repo['composition'].to_numpy(), self.mat_repo['sample_index'].to_numpy()
    def _get_mat_data(self, data_type):
        repo = self.mat_repo.copy()
        # print('data_type to get:', data_type)
        # display(self.mat_repo)
        idx = [i for i in range(repo.shape[0]) if (repo[data_type].iloc[i]) is not None ]
        # print('gotten mat_repo idx:', idx)
        temp_y = repo[data_type].iloc[idx].to_numpy()
        for i in range(len(temp_y)):
            if type(temp_y[i]) is np.ndarray:
                temp_y[i] = temp_y[i].flatten()
            elif isinstance(temp_y[i], list):
                temp_y[i] = np.asarray(temp_y[i]).flatten()
        # print('gotten mat_repo data:', temp_y)
        try:
            y = np.stack( temp_y, axis=0 ).squeeze()
        except:
            print('FAILED TEMP_Y:')
            for i in range(len(temp_y)):
                print(temp_y[i])
        temp_x = repo['composition'].iloc[idx].to_numpy()
        x = np.stack( temp_x, axis = 0 )
        if len(x.shape) == 1:
            x = x[:,None]
        if len(y.shape) == 1:
            y = y[:,None]
        return (x, y)
    def repositories_unify(self, repositories):
        repo0 = respositories[0]
        for idx, repo in enumerate(repositories):
            if idx > 0:
                repo0 = df_unify(repo0, repositories[idx])
        return repo0
    def update_from_sample_pool_wait_for_target(self, env, agent_ID, target_index, target_composition):
        not_found = not np.sum(self.mat_repo['sample_index'].to_numpy() == target_index)
        while not_found:
            self.update_from_sample_pool()
            
#             lent = self.spm.lent_list
            curr = self.spm.list_sample_pool_items()
            curri = np.asarray([s.sample_index for s in curr])
            currc = [s.composition for s in curr if s.sample_index == target_index]
            currc_i = [s.sample_index for s in curr if s.composition == target_composition]
            
            print(f'{agent_ID} waiting for:{target_index} c:{target_composition}, spm:{currc_i},{target_composition};{target_index},{currc}')
            # print('check if composition is in mat repo:')
            # display(self.mat_repo)
            # print('while loop condition:', np.sum(self.mat_repo['sample_index'].to_numpy() == target_index))
            # print('while alt loop condition:', np.sum(self.mat_repo['composition'].to_numpy() == target_composition))
            self.update_from_sample_pool()
            
            # if a sample with the right composition already exists in the mat_repo, use that one.
            repo_idx = np.argwhere(self.mat_repo['composition'].to_numpy() == target_composition)
            if repo_idx: 
                rpo_sidx = self.mat_repo['sample_index'].to_numpy()
                target_index = rpo_sidx[repo_idx].item()
                print('found in mat_repo, sample_index:', target_index)
                # print(f'returned target_index:{target_index}')
                not_found = False
            # if a different target_index has the right composition in spm, use that one.
            elif currc_i:
                target_index = currc_i[0]
                print(f'returned target_index:{target_index}')
                not_found = False
            else:
                yield env.timeout(1)
        return target_index   
    def update_with_measurement(self, meas_type, target_composition, measurement_result):
        # print('update_with_measurement:', meas_type, target_composition, measurement_result)
        # for sample with index target_index, add its measurement value measurement_result
        #print(type(measurement_result), measurement_result)
        if np.max(measurement_result.shape) <= 1: # if the measurement is a scalar, turn it into a 1x1 matrix
            measurement_result = np.asscalar(measurement_result)
        elif type(measurement_result) is np.ndarray and measurement_result.flatten().shape[0] > 1:
            measurement_result = [measurement_result]
        
        self.update_from_sample_pool()
        compositions_ = self.mat_repo['composition'].to_numpy()
        if np.any(compositions_ == target_composition):
            df_index_to_update = np.where(compositions_ == target_composition)[0].flatten().astype(int)
            if df_index_to_update.shape[0] > 1:
                df_index_to_update = df_index_to_update[0]
            # print('cmp dataframe index:',df_index_to_update)
            self.mat_repo[meas_type].iloc[df_index_to_update] = measurement_result # update measurement value
            self.mat_repo = self.mat_repo.sort_values(by=['sample_index']) # sort msir by sample_index
            redo = False
        else:
            spm_items = self.spm.list_sample_pool_items()
            c_lent = np.asarray(self.spm.lent_list)
            c_spm = np.asarray([s.composition for s in spm_items]).flatten()
            print(f'>>>> {meas_type} waiting for c:{target_composition}, repo:{compositions_.flatten()}, spm:{c_spm}, lent:{c_lent}')
            print(f'>>>> All synthesized:{self.spm.all_synthesized}')
            print(f'forcing synth of {target_composition}')
            
            max_item_index = self.spm.get_sample_index_max()
            new_sample_index = max_item_index
            Sample = namedtuple('Sample', 'sample_index, composition, structure, funcprop, status')
            self.spm.sample_pool_append(Sample(new_sample_index, target_composition, 0, 0, ''))
            yield self.env.timeout(1)
            self.update_from_sample_pool()
            redo = True
        yield self.env.timeout(1)
        return redo
    def update_from_sample_pool(self):
        # update cr with samples in sample_pool
        store_items = self.spm.list_sample_pool_items()
        compositions_pool = np.asarray([s.composition for s in store_items]).flatten()[:,None]
        temp = self.mat_repo['composition'].to_numpy()
        compositions_repo = np.stack(temp, axis=0 )
        
        if len(compositions_repo.shape) == 1: # if only one entry in the sample_pool, add an extra empty dimension
            compositions_repo = compositions_repo[:,None]
            
        # for each sample in sample_pool, if there is no sample in msir that is within 1E-3 composition, then add it.    
        for i in range(compositions_pool.shape[0]): 
            v = compositions_repo - np.tile(compositions_pool[i,:][None,:],(compositions_repo.shape[0],1))
            d = np.min( np.linalg.norm(v, axis = 1) )
            if d >= 1E-3:
                self.add_mat_sample(store_items[i])      
    def add_mat_sample(self, sample):
        # add new sample to msir
        # takes in a sample and adds its info to self.msir
        temp = pd.DataFrame(data = {'sample_index': [sample.sample_index],'composition': sample.composition, 'structure': [None], 'funcprop': [None]} )
        self.mat_repo = self.mat_repo.append(temp, ignore_index=True) 
        # display(self.mat_repo)
        # print( tabulate(self.mat_repo, headers='keys', ) )
        ''
    
def normalize(v):
    if len(v.shape) == 1:
        nv = (v - v.min())/(v.max()-v.min())
    else:
        nv = v.copy()
        for i in range(v.shape[0]):
            nv[i,:] = (v[i,:] - v[i,:].min())/(v[i,:].max()-v[i,:].min())
    return nv
