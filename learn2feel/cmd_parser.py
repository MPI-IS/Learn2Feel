#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import configargparse
import yaml
from distutils.util import strtobool

def parse_config(argv=None):
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter

    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'PyTorch implementation of surface-perception learning model'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description)
    parser.add_argument('-c', '--config', required=True,
                        is_config_file=True,
                        help='config file')

    # path parameters
    parser.add_argument('--data_path',type=str,
                        help='Path to the file containing the data.')
    parser.add_argument('--output_path',type=str,
                        help='Path to where the runs should be stored.')

    # training parameters
    parser.add_argument('--epochs',type=int,default=200,help='Training epochs.')
    parser.add_argument('--batch_size',type=int,default=180,help='The batch size.')
    parser.add_argument('--stop_tol',type=float,default=1e-5)
    parser.add_argument('--test_interval',type=int,default=10)
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--fold_split_method',default='all-in-each',
                        choices=['all-in-each','random','stratified',
                                 'stratified-random','subject'])

    # loss parameters
    parser.add_argument('-pdp','--probability_distance_params',type=yaml.safe_load,
                        default={'name':'SinkhornDistance','eps':.1,'max_iter':50,'p':1},
                        help='Distance function between point clouds with parameters')
    parser.add_argument('--soft_rank_reg_val',type=float,default=.1,
                        help="Strength of soft-rank regularization")
    parser.add_argument('--soft_rank_reg_type',default='l2',choices=['l2','kl','log_kl'],
                        help="Type of regularization for soft ranking.")

    # optimizer parameters
    parser.add_argument('--lr', type=float,default=1e-3,help='The learning rate for the algorithm')
    parser.add_argument('--weight_decay',type=float,default=0)

    # dataset parameters
    parser.add_argument('--sensor',default='ft',choices=['ft','accel','both'],
                        help='Use features from force/torque sensor (ft), \
                            accelerometer (accel), or both (both).')
    parser.add_argument('--include_tap',default=True,type=lambda x: bool(strtobool(x)),help='Include tap segments.')
    parser.add_argument('--normalize_tap',default=True,type=lambda x: bool(strtobool(x)),help='Mean-center the tap feature.')
    parser.add_argument('--include_action',default=True,type=lambda x: bool(strtobool(x)),help='Include force and velocity data for slide segments.')
    parser.add_argument('--include_spread',default=False,type=lambda x: bool(strtobool(x)))
    parser.add_argument('--subject_ID',type=int,default=None,
                        help='Subject ID to learn a subject-specific model')
    parser.add_argument('--marginal_weighting',type=str,default='uniform',
                        help='Method used to create the marginals for each ' 
                        'interaction.')
    
    # model architecture parameters
    parser.add_argument('--init_model_path',type=str,default=None,
                        help='Continue training model at this path instead of learning new.')
    parser.add_argument('--model_output_size',type=int,default=3,
                        help='Embedding dimensionality for mapping output.')
    parser.add_argument('--model_seed',type=int,default=None)
    parser.add_argument('--model_hidden_dims', type=int,nargs='*',default=None)
    parser.add_argument('--model_activation', type=str, default=None,
                        choices=[None,'relu','leakyrelu','tanh','sigmoid'])
    parser.add_argument('--model_regularization_methods',type=str,
                        nargs='*',choices=['batchnorm','dropout'], default=None)

    args = parser.parse_args(argv)

    # Checker
    assert args.test_interval <= args.epochs
    if args.fold_split_method == 'all-in-each': assert args.subject_ID is None
    if args.fold_split_method == 'subject': assert args.subject_ID is None
    else: assert args.folds > 2
    
    return args
