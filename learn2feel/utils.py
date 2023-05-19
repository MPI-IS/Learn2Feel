import os
import os.path
import torch
import numpy as np
import pandas as pd

def load_model(config):
    """
    Load model and configuration stored at the path provided in the 
    configuration argument 'init_model_path'.
    """
    if os.path.isfile(config.init_model_path):
        model_save = torch.load(config.init_model_path,map_location='cpu')
        init_state = model_save['model'].state_dict()
        model_config = model_save['config']
        indices = model_save['splits']
        config.sensor = model_config.sensor
        config.include_action = model_config.include_action
        config.include_tap = model_config.include_tap
        config.normalize_tap = model_config.normalize_tap
        config.include_spread = model_config.include_spread
        config.model_output_size = model_config.model_output_size
        config.model_hidden_dims = model_config.model_hidden_dims
        config.model_activation = model_config.model_activation
        config.model_regularization_methods = model_config.model_regularization_methods
    else: 
        raise ValueError(f'The model file {config.init_model_path} does not exist.')
    return init_state, indices

def h5toDataframe(hf, subject_ID = None):
    """Convert the hdf5 data structure into pandas dataframe."""
    # if only training one subject, remove others from dataframe
    if subject_ID is not None:
        if f'Subject {subject_ID}' not in hf.keys():
            raise ValueError('Subject {subject_ID} not in dataset.')  
    df_list = []
    for grp in hf:
        if subject_ID == None: pass
        elif grp != f'Subject {subject_ID}': continue
        for subgrp in hf[grp]:
            new_row = pd.DataFrame({**{'Subject':int(float(grp[-2:])),
                                    'trial number':int(float(subgrp[-2:])),
                                    'surf_l':hf[grp][subgrp].attrs['surf_l'],
                                    'surf_r':hf[grp][subgrp].attrs['surf_r'],
                                    'rating':hf[grp][subgrp].attrs['rating']
                                    },
                                    **{ft:[hf[grp][subgrp][ft][:]] for ft in hf[grp][subgrp]}
                                    })
            df_list.append(new_row)
    return pd.concat(df_list,ignore_index=True)

def group_indices(dataset,folds=5,method='subject'):
    """Split the dataset into different folds for training, cross-validation,
    and testing. 

    :param dataset: dataset class instance
    :param int folds: number of folds to create (ignored if method='subject'), defaults to 5

    :param str method: method used to separate samples into different folds, 
        defaults to 'subject'. Options are: 

         | *all-in-each*: ensure every surface pair is in each fold (not valid for single subject or too many folds).
         | *random*: randomly sort trials.
         | *stratified*: try to match rating distributions across folds.
         | *stratified random*: same as above but randomly.
         | *subject*: each fold is a single subject.
    :raises ValueError: method is not a valid selection
    :return dict: dictionary of indices in each fold for each subject, number of folds
    """
    surfaces = np.unique(dataset.ds['surf_r'])
    pair_sets = []
    if method=='all-in-each':
        for i in range(folds):
            pair_set=[]
            for j in range(len(surfaces)-1):
                ix1 = j%(9-i)
                ix2 = (j+i+1) if len(surfaces)>(j+i+1) else j+1
                s1,s2 = sorted([surfaces[ix1],surfaces[ix2]])
                pair_set.append(f'{s1}_{s2}')
            pair_sets.append(pair_set)
        pair_sets = np.array(pair_sets)
        subject_indices = {s_ix:[[] for _ in range(folds)] \
                           for s_ix in dataset.ds['Subject'].unique()}
        for i in range(len(dataset)):
            s1,s2=np.sort(dataset.ds.iloc[i][['surf_r','surf_l']])
            set_ix = np.where(f'{s1}_{s2}'==pair_sets)[0].item()
            subject_indices[dataset.ds.iloc[i]['Subject']][set_ix].append(i)
            
    elif method=='random':
        subject_indices = {}
        for s_id in np.unique(dataset.ds['Subject']):
            min_ix = dataset.ds[dataset.ds['Subject']==s_id].index.min()
            max_ix = dataset.ds[dataset.ds['Subject']==s_id].index.max()+1
            ixs = np.array_split(np.random.permutation(range(min_ix,max_ix)),folds)
            subject_indices[s_id]=np.array(ixs, dtype="object")
            
    elif 'stratified' in method:
        subject_indices={}
        for s_id in np.unique(dataset.ds['Subject']):
            fold_ix = 0
            subject_indices[s_id]=[np.array([],dtype=int) for _ in range(folds)]
            rating_df = dataset.ds[dataset.ds['Subject']==s_id]['rating']
            for rating in np.unique(rating_df):
                rat_ix = np.where(rating_df==rating.item())[0]+rating_df.index.min()
                if 'random' in method: 
                    np.random.shuffle(rat_ix)
                for ix in rat_ix:
                    subject_indices[s_id][fold_ix] = np.append(subject_indices[s_id][fold_ix],ix)
                    fold_ix = (fold_ix+1) % folds
                    
    elif 'subject' in method:
        folds = len(np.unique(dataset.ds['Subject']))
        subject_indices={1:[]}
        for s_id in np.unique(dataset.ds['Subject']):
            min_ix = dataset.ds[dataset.ds['Subject']==s_id].index.min()
            max_ix = dataset.ds[dataset.ds['Subject']==s_id].index.max()+1
            subject_indices[1].append(np.array(range(min_ix,max_ix),dtype=object))
    
    else:
        raise ValueError(f'{method} is not a valid method for fold creation.')

    return subject_indices,folds

def make_split(index_dict,valid_fold,test_fold):
    """
    Given sample index list from :func:`~utils.group_indices`, and 
    which folds to use for validation and testing, create full samples
    index lists for training, validation, and testing.
    """
    train_ixs = []
    test_ixs = []
    val_ixs = []
    for sub_ix_list in index_dict.values():
        for i,grp in enumerate(sub_ix_list):
            if i==valid_fold:
                val_ixs.extend(grp)
            elif i==test_fold:
                test_ixs.extend(grp)
            else:
                train_ixs.extend(grp)
    return train_ixs,val_ixs,test_ixs

def summarize_run_data(result_dict,val_fold,test_fold):
    '''Summarize and convert loss values to interpretable spearmans 
    correlations. 

    :param dict result_dict: A fully defined :py:attr:`result_dict<learn2feel.training.training_wrapper.result_dict>` 
        from training, validation, and testing. 
    :param int val_fold: Fold index of the validation set used for this run.
    :param int test_fold: Fold index of the test set used for this run.

    '''
    train_rho = [1.0 - result_dict['train_loss']]
    val_rho = [1.0 - result_dict['val_loss']]
    test_rho = [1.0 - result_dict['test_loss']]
    if len(result_dict['train_detailed'].items())>1:
        subject = ['all']
        for sub_id in result_dict['train_detailed'].keys():
            train_rho.extend([1.0-result_dict['train_detailed'][sub_id]])
            val_rho.extend([1.0-result_dict['val_detailed'][sub_id]])
            test_rho.extend([1.0-result_dict['test_detailed'][sub_id]])
            subject.extend([sub_id])
    else:
        subject = list(result_dict['train_detailed'].keys())

    summary = {'Subject': subject,
               'Training Rho': train_rho,
               'Validation fold': val_fold,
               'Validation Rho': val_rho,
               'Testing fold': test_fold,
               'Testing Rho': test_rho}
    
    return pd.DataFrame(summary)
