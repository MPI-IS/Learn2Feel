import os
import os.path as osp

import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader,SubsetRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter

from learn2feel.cmd_parser import parse_config
import learn2feel.training as training
import learn2feel.dataset as dataset
import learn2feel.losses as losses
import learn2feel.distances as distances
import learn2feel.models as models
import learn2feel.utils as utils


def main(config: argparse.Namespace):
    # put on gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load existing model and set torch seed
    if config.init_model_path is not None:
        print(f"Loading model at {config.init_model_path}. "
              "Loading model and parameters configuration.")
        init_state, _ = utils.load_model(config)
    if config.model_seed is None:
        config.model_seed = np.random.randint(1000)
    torch.random.manual_seed(config.model_seed)

    # Setup folders and save config
    os.makedirs(config.output_path,exist_ok=True)
    with open(os.path.join(config.output_path,'conf.yaml'), 'w') as conf_file:
        yaml.dump(config, conf_file)

    #setup dataset and create folds
    surface_pairs = dataset.surface_pair_perception(**vars(config))
    fold_indexes,splits = utils.group_indices(surface_pairs,
                                                config.folds,
                                                config.fold_split_method)
    
    #loss
    criterion = losses.spearmans_rank(config.soft_rank_reg_val,
                                      config.soft_rank_reg_type)
    distance_func = distances.distance_parser(config.probability_distance_params)
    #loop through all combinations of folds

    result_df = []
    for test_ix in range(splits):
        for val_ix in range(splits):
            if test_ix==val_ix: continue
            print(f'Training model with validation fold {val_ix} '
                  f'and test fold {test_ix}.')
            torch.manual_seed(config.model_seed)

            #create tracker
            split_id = osp.join(config.output_path,f'val{val_ix}_test{test_ix}')
            writer = SummaryWriter(log_dir=split_id)

            #create list of indices for train, validation, and test based on folds
            indices={}
            indices['train'],\
                indices['validation'],\
                    indices['test'] = utils.make_split(fold_indexes,val_ix,test_ix)

            #create the data standardization using training samples
            #if (config.fold_split_method != 'subject'):
            surface_pairs.set_transforms(indices=indices['train'])

            #samplers and loaders for each set based on fold indices
            sampler={};loader={}
            for split_name in ['train','validation','test']:
                sampler[split_name] = SubsetRandomSampler(indices[split_name])
                loader[split_name] = DataLoader(surface_pairs,
                                                batch_size=config.batch_size,
                                                collate_fn=dataset.collate_fn,
                                                sampler=sampler[split_name],
                                                pin_memory=True)

            #build model and load the state we previously loaded if exists
            model = models.fc_mapping_function(input_dim=surface_pairs.get_dim(),
                                               hidden_dims=config.model_hidden_dims,
                                               output_dim=config.model_output_size,
                                               activation=config.model_activation,
                                               regularizations=config.model_regularization_methods)
            if config.init_model_path is not None:
                model.load_state_dict(init_state) # type: ignore
            model.to(device)
            optimizer = optim.Adam(model.parameters(),lr=config.lr,
                                weight_decay=config.weight_decay)
            
            #train the model 
            train_wrapper = training.training_wrapper(model,optimizer,
                                                      distance_func,
                                                      criterion,device,
                                                      writer)
            train_wrapper.train(loader,config.epochs,config.test_interval)
            test_loss,test_detail = train_wrapper.evaluate(loader['test'])
            train_wrapper.populate_results('test',test_loss,test_detail)
            writer.close()
            
            result_dict = train_wrapper.result_dict
            result_dict['splits']=indices
            result_dict['config']=config
            torch.save(result_dict,f'{split_id}/model')
            
            # Save results summary
            result_df.append(utils.summarize_run_data(result_dict,val_ix,test_ix))
            pd.concat(result_df,ignore_index=True).to_csv(osp.join(config.output_path,'summary.csv'))

def launcher():
    config = parse_config()
    main(config)

if __name__ == "__main__":
    config = parse_config()
    main(config)

# python -m learn2feel.main -c configs/config.yaml