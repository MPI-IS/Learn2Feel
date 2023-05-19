import h5py
import os.path
import numpy as np
import numpy.ma as ma
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from itertools import chain
from torchvision import transforms
import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from typing import Optional

import learn2feel.utils as utils

class surface_pair_perception(Dataset):
    """ Dataset class that manages a pandas dataframe containing data 
    from the Learn2Feel dataset.

    When returning an element, marginal probability densities are 
    created for each left and right-handed data sequence at the 
    specified index using :func:`dataset.define_marginals`.

    :param str data_path: Path to the data.
    :param bool include_tap: Include tap spectral centroid feature.
    :param bool normalize_tap: Mean-center tap features.
    :param bool include_action: Include force and velocity.
    :param bool include_spread: Include distribution spread feature.
    :param str sensor: Use only data from the force/torque sensor ('ft'), 
        accelerometer ('accel'), or both ('both').
    :param list,optional train_indices: Indices of samples used to compute 
        transforms.
    :param int,optional subject_ID: Only use samples from this 
        particular subject.
    :param str marginal_weighting: Method for defining marginals, 
        defaults to 'uniform'.
    :raises ValueError: Subject ID not in dataset.
    :.__getitem__() returns: 

         | (*Tensor*, :math:`(P_1, D)`) right hand features, 
         | (*Tensor*, :math:`(P_1, 1)`) right hand marginal, 
         | (*Tensor*, :math:`(P_2, D)`) left handed features, 
         | (*Tensor*, :math:`(P_2, 1)`) left handed marginal, 
         | (*int*) similarity rating, 
         | (*dict*) surface names, and 
         | (*int*) subject ID.
    """
    def __init__(self,
                 data_path: str,
                 include_tap: bool = True,
                 normalize_tap: bool = False,
                 include_action: bool = True,
                 include_spread: bool = False,
                 sensor: str = 'both',
                 train_indices: Optional[list] = None,
                 subject_ID: Optional[int] = None,
                 marginal_weighting: str = 'uniform',
                 **_,
                 ):
        
        self.marginal_weighting = marginal_weighting
        self.include_tap = include_tap
        if os.path.isfile(data_path):
            hf = h5py.File(data_path,'r')
        else:   
            raise ValueError('Data not found. Please provide correct path to data.')

        self.ds = utils.h5toDataframe(hf, subject_ID)
        
        # add features/columns that we will take from dataframe
        feat_set = ['friction']
        if (sensor=='ft') or (sensor=='both'): 
            feat_set.extend(['power','slide_spectral_centroid'])
            if include_spread: feat_set.extend(['spread'])
        if (sensor=='accel') or (sensor=='both'):
            feat_set.extend(['power_acc','slide_spectral_centroid_acc'])
            if include_spread: feat_set.extend(['spread_acc'])
        if include_action: feat_set.extend(['vel','force']) 
        if include_tap: feat_set.extend(['tap_spectral_centroid']) 
        self.features = feat_set
    
        self.separate = not normalize_tap

        self.set_transforms(indices=train_indices)
            
    def __getitem__(self,index: int):
        row = self.ds.iloc[index]
        subject = row['Subject']
        points={}
        surfaces={}
        for hand in ['r','l']:
            a0 = row[f'{self.features[0]}_{hand}'].shape[0]
            if self.include_tap: a0 += row.filter(regex=(f"tap.*_{hand}")).item().shape[0]
            data_array = ma.masked_array(np.zeros([a0,len(self.features)]),mask=True)
            
            for i,feat in enumerate(self.features):
                arr = row[f'{feat}_{hand}'].flatten()
                if 'tap' in feat:
                    data_array[a0-arr.shape[0]:,i] = arr  # Replace '-arr.shape:' to 'a0-arr.shape:' to avoid no tap error
                else:
                    data_array[:arr.shape[0],i] = arr
            if self.transform_dict[subject] is not None:
                sub = self.transform_dict[subject][0]
                div = self.transform_dict[subject][0]
                tf = transforms.Compose([transforms.Lambda(lambda x: (x-sub)/div)])
                data_array = tf(data_array)     
            points[hand] = torch.tensor(data_array,dtype=torch.float32)
            surfaces[hand] = row[f'surf_{hand}']
        label = torch.tensor((9.0 - row['rating'].item())/8.0)

        # compute marginals
        # Make modifications here if you want to implement different 
        # margin definitions. Add parameters (e.g. subject) to the 
        # function call if necessary. 
        margins={}
        margins['r'],margins['l'] = define_marginals(points['r'],points['l'],
                                                     self.marginal_weighting)

        return points['r'],margins['r'],\
            points['l'],margins['l'],label,surfaces,subject
    
    def __len__(self) -> int:
        return self.ds.shape[0]
    
    def set_transforms(self,indices: Optional[list] = None):
        """Compute the mean and standard deviation of each feature in
        the dataset on a per-subject basis. Store the transformations
        and apply them in :meth:`__getitem__`.

        :param indices: Indices of the dataset samples that are used to 
            compute normalization. If *None*, learn transform on full 
            dataset, defaults to None.
        :type indices: list, optional
        """
        transform_dict={}
        hand=['r','l']
        tf_ds = self.ds.iloc[sorted(indices)] if indices else self.ds
        # create different transform for each subject
        for sub in tf_ds['Subject'].unique():
            sub_df = tf_ds[tf_ds['Subject']==sub]
            subtract_arr = np.zeros(len(self.features))
            div_arr = np.ones(len(self.features))
            for i,feat in enumerate(self.features):
                dat_arr = np.concatenate([list(chain.from_iterable(sub_df[f'{feat}_{hd}'])) for hd in hand])
                subtract_arr[i] = dat_arr.mean()
                div_arr[i] = dat_arr.std()
                
                if ('tap' in feat) and (self.separate):
                    subtract_arr[i] = 0.0        
                    div_arr[i] = 1.0

            transform_dict[sub] = [subtract_arr,div_arr]
        self.transform_dict=transform_dict

    def get_dim(self):
        return len(self.features)

def collate_fn(batch: list):
    """Reconfigure a batch to account for different length samples. 
    Surface similarity ratings, surface labels, and subject IDs are 
    concatenated normally. 

    :return: Padded sequences/marginals from right and left hands, labels,
        subject list, and surface pair list.
    :rtype: list[Tensor, Tensor], list[Tensor, Tensor], Tensor, Tensor, list
    """
    #data[0] right data
    #data[1] right marginals
    #data[2] left data
    #data[3] left marginals
    #data[4] rating
    #data[5] surfaces
    #data[6] subject
    data_r = []
    margin_r = []
    data_l = []
    margin_l = []
    lbl=[];sf=[];subj=[]
    
    for data in batch:
        # Ignore sample if it has no data
        if (data[0].shape[0]==0) or (data[2].shape[0]==0):
            continue    
        if data[0].shape[0]==1:
            data_r.append(torch.cat((data[0],data[0])))
            margin_r.append(torch.cat((data[1],data[1])))
        else:
            data_r.append(data[0])
            margin_r.append(data[1])
        if data[2].shape[0]==1:
            data_l.append(torch.cat((data[2],data[2])))
            margin_l.append(torch.cat((data[3],data[3])))
        else:
            data_l.append(data[2])
            margin_l.append(data[3])
        lbl.append(data[4])
        sf.append(data[5])
        subj.append(data[6])
    lbl = torch.tensor(lbl,dtype=torch.float32)
    subj = torch.tensor(subj,dtype=torch.float32)
    
    #pad and create marginals
    dr_pad = [pad_sequence(data_r,batch_first=True),
              pad_sequence(margin_r,batch_first=True)]
    dl_pad = [pad_sequence(data_l,batch_first=True),
              pad_sequence(margin_l,batch_first=True)]
    return dr_pad,dl_pad,lbl,subj,sf

 
def define_marginals(data_right: torch.Tensor, 
                     data_left: torch.Tensor, 
                     weighting_method: str = 'uniform'):
    """Create marginals for the right and left-handed data of a sequence.
    Uniform will give each segment of the r/l sequence equal weight. 

    (*New margin definition functions can be inserted/linked here.*)

    :param torch.Tensor data_right: segments from the right-handed interaction
    :param torch.Tensor data_left: segments from the left-handed interaction
    :param str weighting_method: weighting method used to define marginals
    :raises NotImplementedError: Weighting method for defining marginals 
        must be defined.
    :return: right and left-handed marginals
    :rtype: Tensor, Tensor
    """
    if weighting_method == 'uniform':
        r_shape = data_right.shape[0]
        l_shape = data_left.shape[0]
        mgn_r = torch.ones(r_shape,requires_grad=False)/r_shape
        mgn_l = torch.ones(l_shape,requires_grad=False)/l_shape
    else: raise NotImplementedError
    return mgn_r,mgn_l