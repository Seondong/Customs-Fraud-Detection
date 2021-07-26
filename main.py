import logging
import argparse
import os
import pathlib
import csv
import pdb
import time
import pickle
import warnings
import dataset
import sys
import math
from itertools import islice
from collections import defaultdict
from datetime import timedelta
import datetime
import numpy as np
import torch
import torch.utils.data as Data
import pandas as pd
import torch
from model.AttTreeEmbedding import Attention, DATEModel
from ranger import Ranger
from utils import torch_threshold, metrics, metrics_active
from query_strategies import uncertainty, random
warnings.filterwarnings("ignore")


def make_logger(curr_time, name=None):
    """ Initialize loggers, log files are saved under the ./intermediary/logs directory 
        ToDo: Change all print functions to logger.   
    """
    
    # five levels of logging: DEBUG, INFO, WARNING, ERROR, CRITICAL (from mild to severe)
    logger = logging.getLogger(name)
    
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter("%(message)s")
    
    console = logging.StreamHandler()
    file_handler = logging.FileHandler(filename="./intermediary/logs/log_"+curr_time+".log")
    
    console.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)
    
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console)
    logger.addHandler(file_handler)
    
    return logger


def evaluate_inspection(chosen_rev,chosen_cls,xgb_testy,revenue_test):
    """ Evaluate the model """
    precisions, recalls, f1s, revenues = metrics_active(chosen_rev,chosen_cls,xgb_testy,revenue_test)
    best_score = f1s
    return precisions, recalls, f1s, revenues


def inspection_plan(rate_init, rate_final, numWeeks, option):
    """ Inspection plan for next n weeks """
    if option == 'direct_decay':
        return np.linspace(rate_final, rate_final, numWeeks)
    
    if option == 'linear_decay':
        return np.linspace(rate_init, rate_final, numWeeks)
    
    if option == 'fast_linear_decay':
        first_half = np.linspace(rate_init, rate_final, 10)
        second_half = np.linspace(rate_final, rate_final, numWeeks - len(first_half))
        return np.concatenate((first_half, second_half))


# Selection stragies
def initialize_sampler(samp, args):
    """Initialize selection strategies"""
    if samp == 'random':
        from query_strategies import random;
        sampler = random.RandomSampling(args)
    elif samp == 'xgb':
        from query_strategies import xgb;
        sampler = xgb.XGBSampling(args)
    elif samp == 'xgb_lr':
        from query_strategies import xgb_lr;
        sampler = xgb_lr.XGBLRSampling(args)
    elif samp == 'badge':
        from query_strategies import badge;
        sampler = badge.BadgeSampling(args)
    elif samp in ['DATE', 'noupDATE', 'randomupDATE']:
        from query_strategies import DATE;
        sampler = DATE.DATESampling(args)
    elif samp == 'diversity':
        from query_strategies import diversity;
        sampler = diversity.DiversitySampling(args)
    elif samp == 'bATE':
        from query_strategies import bATE;
        sampler = bATE.bATESampling(args)
    elif samp == 'upDATE':
        from query_strategies import upDATE;
        sampler = upDATE.upDATESampling(args)
    elif samp == 'gATE':
        from query_strategies import gATE;
        sampler = gATE.gATESampling(args)
    elif samp == 'ssl_ae':
        from query_strategies import ssl_ae;
        sampler = ssl_ae.SSLAutoencoderSampling(args)
    elif samp == 'tabnet':
        from query_strategies import tabnet;
        sampler = tabnet.TabnetSampling(args)
    elif samp == 'deepSAD': # check
        from query_strategies import deepSAD;
        sampler = deepSAD.deepSADSampling(args)
    elif samp == 'multideepSAD':
        from query_strategies import multideepSAD;
        sampler = multideepSAD.multideepSADSampling(args)
    elif samp == 'hybrid':
        from query_strategies import hybrid;
        sampler = hybrid.HybridSampling(args)
    elif samp == 'pot':
        from query_strategies import pot;
        sampler = pot.POTSampling(args)
    elif samp == 'adahybrid':
        from query_strategies import adahybrid;
        sampler = adahybrid.AdaHybridSampling(args)
    elif samp == 'pvalue':
        from query_strategies import p_value;
        sampler = p_value.pvalueSampling(args)
    else:
        sampler = None
        print('Make sure the sampling strategy is listed in the argument --sampling')
    return sampler


if __name__ == '__main__':
    
    curr_time = str(round(time.time(),3))
    print('Experiment starts: ', curr_time)
    
    # Initiate directories
    pathlib.Path('./results').mkdir(parents=True, exist_ok=True) 
    pathlib.Path('./results/performances').mkdir(parents=True, exist_ok=True)
    pathlib.Path('./results/ada_ratios').mkdir(parents=True, exist_ok=True)    
    pathlib.Path('./results/query_indices').mkdir(parents=True, exist_ok=True)
    pathlib.Path('./intermediary').mkdir(parents=True, exist_ok=True)
    pathlib.Path('./intermediary/saved_models').mkdir(parents=True, exist_ok=True)
    pathlib.Path('./intermediary/logs').mkdir(parents=True, exist_ok=True)
    pathlib.Path('./intermediary/xgb_models').mkdir(parents=True, exist_ok=True)
    pathlib.Path('./intermediary/tn_models').mkdir(parents=True, exist_ok=True)
    pathlib.Path('./intermediary/torch_data').mkdir(parents=True, exist_ok=True)
    pathlib.Path('./intermediary/leaf_indices').mkdir(parents=True, exist_ok=True)
    pathlib.Path('./intermediary/embeddings').mkdir(parents=True, exist_ok=True)
    pathlib.Path('./uncertainty_models').mkdir(parents=True, exist_ok=True)
    pathlib.Path('./temp').mkdir(parents=True, exist_ok=True)
    
    logger = make_logger(curr_time)
    
    # Parse argument
    parser = argparse.ArgumentParser()
    
    # Hyperparameters related to DATE
    parser.add_argument('--epoch', type=int, default=20, help="Number of epochs for DATE-related models")
    parser.add_argument('--batch_size', type=int, default=10000, help="Batch size for DATE-related models")
    parser.add_argument('--dim', type=int, default=16, help="Hidden layer dimension")
    parser.add_argument('--lr', type=float, default=0.005, help="learning rate")
    parser.add_argument('--l2', type=float, default=0.01, help="l2 reg")
    parser.add_argument('--alpha', type=float, default=10, help="Regression loss weight")
    parser.add_argument('--head_num', type=int, default=4, help="Number of heads for self attention")
    parser.add_argument('--use_self', type=int, default=1, help="Whether to use self attention")
    parser.add_argument('--fusion', type=str, choices=["concat","attention"], default="concat", help="Fusion method for final embedding")
    parser.add_argument('--agg', type=str, choices=["sum","max","mean"], default="sum", help="Aggreate type for leaf embedding")
    parser.add_argument('--act', type=str, choices=["mish","relu"], default="relu", help="Activation function")
    
    # Hyperparameters related to customs selection
    parser.add_argument('--prefix', type=str, default='results', help="experiment name used as prefix for results file")
    parser.add_argument('--initial_masking', type=str, default="random", choices = ['random', 'importer', 'natural'], help="Masking some initial training data for simulating partially labeled scenario (for synthetic and m, n, t dataset)")
    parser.add_argument('--devices', type=str, default=['0','1','2','3'], help="list of gpu available")
    parser.add_argument('--device', type=str, default='0', help='select which device to run, choose gpu number in your devices or cpu') 
    parser.add_argument('--output', type=str, default="result"+"-"+curr_time, help="Name of output file")
    parser.add_argument('--sampling', type=str, default = 'bATE', choices=['random', 'xgb', 'xgb_lr', 'DATE', 'diversity', 'badge', 'bATE', 'upDATE', 'gATE', 'hybrid', 'adahybrid', 'tabnet', 'ssl_ae', 'noupDATE', 'randomupDATE', 'deepSAD', 'multideepSAD', 'pot', 'pvalue'], help='Sampling strategy')
    parser.add_argument('--initial_inspection_rate', type=float, default=100, help='Initial inspection rate in training data by percentile')
    parser.add_argument('--final_inspection_rate', type=float, default = 5, help='Percentage of test data need to query')
    parser.add_argument('--inspection_plan', type=str, default = 'direct_decay', choices=['direct_decay','linear_decay','fast_linear_decay'], help='Inspection rate decaying option for simulation time')
    parser.add_argument('--mode', type=str, default = 'finetune', choices = ['finetune', 'scratch'], help = 'finetune last model or train from scratch')
    parser.add_argument('--subsamplings', type=str, default = 'xgb/random', help = 'available for hybrid sampling, the list of sub-sampling techniques seperated by /')
    parser.add_argument('--weights', type=str, default = '0.9/0.1', help = 'available for hybrid sampling, the list of weights for sub-sampling techniques seperated by /')
    parser.add_argument('--uncertainty', type=str, default = 'naive', choices = ['naive', 'self-supervised'], help = 'Uncertainty principle : ambiguity of illicitness or self-supervised manner prediction')
    parser.add_argument('--rev_func', type=str, default = 'log', choices = ['log'], help = 'Uncertainty principle : ambiguity of illicitness or self-supervised manner prediction')
    parser.add_argument('--closs', type=str, default = 'bce', choices = ['bce', 'focal'], help = 'Classification loss function')
    parser.add_argument('--rloss', type=str, default = 'full', choices = ['full', 'masked'], help = 'Regression loss function')
    parser.add_argument('--train_from', type=str, default = '20130101', help = 'Training period start from (YYYYMMDD)')
    parser.add_argument('--test_from', type=str, default = '20130201', help = 'Testing period start from (YYYYMMDD)')
    parser.add_argument('--test_length', type=int, default=7, help='Single testing period length (e.g., 7)')
    parser.add_argument('--valid_length', type=int, default=7, help='Validation period length (e.g., 7)')
    parser.add_argument('--data', type=str, default='synthetic', choices = ['synthetic', 'synthetic-k', 'synthetic-k-partial', 'real-n', 'real-m', 'real-t', 'real-c'], help = 'Dataset')
    parser.add_argument('--numweeks', type=int, default=50, help='number of test weeks (week if test_length = 7)')
    # parser.add_argument('--semi_supervised', type=int, default=0, help='Additionally using uninspected, unlabeled data (1=semi-supervised, 0=fully-supervised)')
    parser.add_argument('--identifier', type=str, default=curr_time, help='identifier for each execution')
    parser.add_argument('--save', type=int, default=0, help='Save intermediary files (1=save, 0=not save)')

    # Ada hyperparameters:
    parser.add_argument('--ada_lr', type=float, default=0.8, help="learning rate for adahybrid")
    parser.add_argument('--num_arms', type=int, default=21, help="number of arms for adahybrid")

    # Arguments
    args = parser.parse_args()
    epochs = args.epoch
    dim = args.dim
    lr = args.lr
    weight_decay = args.l2
    head_num = args.head_num
    act = args.act
    fusion = args.fusion
    alpha = args.alpha
    use_self = args.use_self
    agg = args.agg
    samp = args.sampling
    initial_inspection_rate = args.initial_inspection_rate
    final_inspection_rate = args.final_inspection_rate
    inspection_rate_option = args.inspection_plan
    mode = args.mode
    unc_mode = args.uncertainty
    train_begin = args.train_from 
    test_begin = args.test_from
    test_length = args.test_length
    valid_length = args.valid_length
    chosen_data = args.data
    numWeeks = args.numweeks
    # semi_supervised = args.semi_supervised
    save = args.save
    initial_masking = args.initial_masking
    ada_lr = args.ada_lr
    num_arms = args.num_arms
    
    logger.info(args)
    
    
    # Load datasets 
    if chosen_data == 'synthetic':
        data = dataset.Syntheticdata(path='./data/synthetic-imports-declarations.csv')
    elif chosen_data == 'synthetic-k':
        data = dataset.SyntheticKdata(path='./data/df_syn_ano_0429_merge.csv')  # fully labeled
    elif chosen_data == 'synthetic-k-partial':
        data = dataset.SyntheticKdata(path='./data/df_syn_ano_0429_merge_partially_labeled.csv')   # partially labeled
        args.initial_masking = 'natural'   # since this data is given as partially labeled, it does not need extra label masking.
        initial_masking = 'natural'   
    elif chosen_data == 'real-n':
        data = dataset.Ndata(path='./data/ndata.csv')
    elif chosen_data == 'real-m':
        data = dataset.Mdata(path='./data/mdata.csv')
    elif chosen_data == 'real-t':
        data = dataset.Tdata(path='./data/tdata.csv')
    elif chosen_data == 'real-c':
        data = dataset.Cdata(path='./data/cdata.csv')  
    
        
    # Saving simulation results: Output file will be saved under ./results/performances/ directory
    subsamps = args.subsamplings.replace('/','+')
    if samp not in ['hybrid', 'adahybrid', 'pot', 'pvalue']:
        subsamps = 'single'
        
    # Open files:
    output_file =  "./results/performances/" + args.prefix + '-' + args.output + '-' + chosen_data + '-' + samp + '-' + subsamps + '-' + str(final_inspection_rate) + ".csv"
    with open(output_file, 'a') as ff:
        output_metric_name = ['runID', 'data', 'num_train','num_valid','num_test','num_select','num_inspected','num_uninspected','num_test_illicit','test_illicit_rate', 'upper_bound_precision', 'upper_bound_recall','upper_bound_rev', 'sampling', 'initial_inspection_rate', 'current_inspection_rate', 'final_inspection_rate', 'inspection_rate_option', 'mode', 'subsamplings', 'initial_weights', 'current_weights', 'unc_mode', 'train_start', 'valid_start', 'test_start', 'test_end', 'numWeek', 'precision', 'recall', 'revenue', 'norm-precision', 'norm-recall', 'norm-revenue', 'save']
        print(",".join(output_metric_name),file=ff)
    
    if samp == 'adahybrid':
        weight_file =  "./results/ada_ratios/" + args.prefix + '-' + args.output + '-' + samp + '-' + subsamps + '-' + str(final_inspection_rate) + ".csv"
        with open(weight_file, 'a') as ff:
            output_weight_name = ['runID', 'data', 'sampling', 'subsamplings', 'numWeek', 'norm-precision', 'norm-recall', 'norm-revenue', 'lr'] + [f'{i/(num_arms-1)} explore rate' for i in range(num_arms)] + ['chosen_rate', 'chosen_arm']
            print(",".join(output_weight_name), file=ff)
    path = None
    uncertainty_module = None
    
    # Initial dataset split
    train_start_day = datetime.date(int(train_begin[:4]), int(train_begin[4:6]), int(train_begin[6:8]))
    test_start_day = datetime.date(int(test_begin[:4]), int(test_begin[4:6]), int(test_begin[6:8]))
    test_length = timedelta(days=test_length)    
    test_end_day = test_start_day + test_length
    valid_length = timedelta(days=valid_length)
    valid_start_day = test_start_day - valid_length
    data.split(train_start_day, valid_start_day, test_start_day, test_end_day, valid_length, test_length, args)
    confirmed_inspection_plan = inspection_plan(initial_inspection_rate, final_inspection_rate, numWeeks, inspection_rate_option)
    logger.info('Inspection rate for testing periods: %s', confirmed_inspection_plan)
       
    if samp in ['hybrid', 'adahybrid', 'pot', 'pvalue']:
        subsamplings = args.subsamplings
        initial_weights = [float(weight) for weight in args.weights.split("/")]
        final_weights = initial_weights
    else:
        subsamplings = '-'
        initial_weights = '-'
        final_weights = '-'
                
    # Initialize a sampler (We put it outside the week loop since we do not change sampler every week)
    # NOTE: If you put this inside the week loop, new sampler is initialized every week, which means that parameters in sampler are also initialized)    
    sampler = initialize_sampler(samp, args)      
        
    # Customs selection simulation for long term (if test_length = 7 days, simulate for numWeeks)
    for i in range(numWeeks):
        
        if test_start_day.strftime('%y-%m-%d') > max(data.df["sgd.date"]):
            logger.info('Simulation period is over.')
            logger.info('Terminating ...')
            sys.exit()

        # Feature engineering for train, valid, test data
        data.episode = i
        current_inspection_rate = confirmed_inspection_plan[i]  # ToDo: Add multiple decaying strategy
        logger.info(f'Test episode: #{i}, Current inspection rate: {current_inspection_rate}')

        if samp not in ['random']: 
            data.featureEngineering()
        else:
            data.offset = data.test.index[0]
        
        # Initialize uncertainty module for some cases
        if unc_mode == 'self-supervised':
            if samp in ['bATE', 'diversity', 'hybrid', 'upDATE', 'gATE', 'adahybrid', 'pot', 'pvalue']:
                if uncertainty_module is None :
                    uncertainty_module = uncertainty.Uncertainty(data.train_lab, './uncertainty_models/')
                    uncertainty_module.train()
                uncertainty_module.test_data = data.test 
        
        num_samples = int(len(data.test)*current_inspection_rate/100)
        
        # Retrieve subsampler weights from the previous week, for hybrid models
        if samp in ['hybrid', 'adahybrid', 'pot', 'pvalue']:
            try:
                final_weights = sampler.get_weights()
            except NameError:
                pass  # use the previously defined final_weights (= initial_weights)
  
        # If we need to update sampler every week, you can initialize the sampler here.
        
        # set uncertainty module
        sampler.set_uncertainty_module(uncertainty_module)
        
        # set previous weeks' weights, for hybrid models
        if samp in ['hybrid', 'adahybrid', 'pot', 'pvalue']:
            sampler.set_weights(final_weights)
        
        # set data to sampler
        sampler.set_data(data)
        
        # POT should measure the domain shift just after the data is loaded. 
        if samp == 'pot':
            sampler.update_subsampler_weights()
        elif samp == 'pvalue':
            sampler.update_subsampler_weights()

        try:
            chosen = sampler.query(num_samples)
            
        except:
            import traceback
            traceback.print_exc()
            
        logger.info("--------Evaluating selection results---------")   
        logger.info("# of unique queried item: %s, # of queried item: %s, # of samples to be queried: %s", len(set(chosen)), len(chosen), num_samples)
        try:
            assert len(set(chosen)) == num_samples
        except AssertionError:
            import traceback
            traceback.print_exc()        
 
        # Indices of sampled imports (Considered as fraud by model) -> This will be inspected thus annotated.    
        indices = [point + data.offset for point in chosen]
        
        # Originally, chosen trade should be annotated.
        # Compatible with simulating on synthetic-k-partial dataset. We need this procedure to evaluate the selection strategy on given partially-labeled datasets. 
        indices = data.df['illicit'][indices].notnull().loc[lambda x: x==True].index.values
        
        inspected_imports = data.df.iloc[indices]
        uninspected_imports = data.df.loc[set(data.test.index)-set(inspected_imports.index)]
        uninspected_imports['illicit'] = float('nan')
        uninspected_imports['revenue'] = float('nan')
      
        logger.debug(inspected_imports[:5])
        
        # tune the uncertainty
        if unc_mode == 'self-supervised' and samp in ['bATE', 'diversity', 'hybrid', 'upDATE', 'gATE', 'adahybrid', 'pot', 'pvalue']:
            uncertainty_module.retrain(data.test.iloc[indices - data.offset])
        
        # Evaluation
        active_rev = inspected_imports['revenue']
        active_rev = active_rev.transpose().to_numpy()

        active_cls = inspected_imports['illicit']
        active_cls = active_cls.transpose().to_numpy()

        # Added to handle semi-supervised inputs
        active_cls_notna = active_cls[~np.isnan(active_cls)]
        active_rev_notna = active_rev[~np.isnan(active_rev)]
        illicit_test_notna = data.test_cls_label[~np.isnan(data.test_cls_label)]
        revenue_test_notna = data.test_reg_label[~np.isnan(data.test_reg_label)]

        active_precisions, active_recalls, active_f1s, active_revenues = evaluate_inspection(active_rev_notna, active_cls_notna, illicit_test_notna, revenue_test_notna)
        logger.info(f'Metrics Active DATE:\n Pr@{current_inspection_rate}:{round(active_precisions, 4)}, Re@{current_inspection_rate}:{round(active_recalls, 4)} Rev@{current_inspection_rate}:{round(active_revenues, 4)}') 

        with open(output_file, 'a') as ff:
            upper_bound_precision = min(np.sum(illicit_test_notna)/len(chosen), 1)
            upper_bound_recall = min(len(chosen)/np.sum(illicit_test_notna), 1)
            upper_bound_revenue = min(sum(sorted(revenue_test_notna, reverse=True)[:len(chosen)]) / np.sum(revenue_test_notna), 1)

            norm_precision = active_precisions/upper_bound_precision
            norm_recall = active_recalls/upper_bound_recall
            norm_revenue = active_revenues/upper_bound_revenue
            
            if samp in ['hybrid', 'adahybrid', 'pot', 'pvalue']:
                initial_weights_str = '/'.join([str(weight) for weight in initial_weights])
                final_weights_str = '/'.join([str(weight) for weight in final_weights])

            else:
                initial_weights_str = '-'
                final_weights_str = '-'
            
            output_metric = [curr_time, chosen_data, len(data.train_lab), len(data.valid_lab), len(data.test), len(chosen), len(inspected_imports), len(uninspected_imports), np.sum(data.test_cls_label), np.mean(data.test_cls_label), upper_bound_precision, upper_bound_recall, upper_bound_revenue, samp, initial_inspection_rate, current_inspection_rate, final_inspection_rate, inspection_rate_option, mode, subsamplings, initial_weights_str, final_weights_str, unc_mode, train_start_day.strftime('%y-%m-%d'), valid_start_day.strftime('%y-%m-%d'), test_start_day.strftime('%y-%m-%d'), test_end_day.strftime('%y-%m-%d'), i+1, round(active_precisions,4), round(active_recalls,4), round(active_revenues,4), round(norm_precision,4), round(norm_recall,4), round(norm_revenue,4), save]
                
            output_metric = list(map(str,output_metric))
            logger.debug(output_metric)
            print(",".join(output_metric),file=ff)

        if samp == 'adahybrid':
            with open(weight_file, 'a') as ff:
                subsamplings = args.subsamplings
                weights = '/'.join([str(weight) for weight in final_weights])
                    
                upper_bound_precision = min(np.sum(illicit_test_notna)/len(chosen), 1)
                upper_bound_recall = min(len(chosen)/np.sum(illicit_test_notna), 1)
                upper_bound_revenue = min(sum(sorted(revenue_test_notna, reverse=True)[:len(chosen)]) / np.sum(revenue_test_notna), 1)

                norm_precision = active_precisions/upper_bound_precision
                norm_recall = active_recalls/upper_bound_recall
                norm_revenue = active_revenues/upper_bound_revenue
                
                
                output_metric = [curr_time, chosen_data, samp, subsamplings, i+1, round(norm_precision,4), round(norm_recall,4), round(norm_revenue,4), ada_lr] + list(sampler.weight_sampler.p) + [sampler.weight_sampler.value, sampler.weight_sampler.arm]
                    
                output_metric = list(map(str,output_metric))
                logger.debug(output_metric)
                print(",".join(output_metric),file=ff)
        
        output_file_indices =  "./results/query_indices/" + curr_time + '-' + samp + '-' + subsamplings.replace('/','+') + '-' + str(current_inspection_rate) + '-' + mode + "-week-" + str(i) + ".csv"
            
        with open(output_file_indices, "w", newline='') as queryFiles:
            wr = csv.writer(queryFiles, delimiter = ",")
            wr.writerow(['Experiment ID', curr_time])
            wr.writerow(['Dataset', chosen_data])
            wr.writerow(['Episode', i])
            wr.writerow(['Test_start_day', test_start_day])
            wr.writerow(['Test_end_day', test_end_day])
            
            if samp in ['hybrid', 'adahybrid', 'pot', 'pvalue']:
                tmpIdx = 0
                for subsampler, num in zip(args.subsamplings.split('/'), sampler.ks):
                    row = [subsampler]
                    row.extend(indices[tmpIdx:tmpIdx+num])
                    wr.writerow(row)
                    tmpIdx += num
            else:
                row = [samp]
                row.extend(indices)
                wr.writerow(row)
        
        # Review needed: Check if the weights are updated as desired.
        if samp == 'adahybrid':
            sampler.update_subsampler_weights(norm_precision)

        # Renew valid & test period & dataset
        if i == numWeeks - 1:
            logger.info('Simulation period is over.')
            logger.info('Terminating ...')
            sys.exit()
            
        test_start_day = test_end_day
        test_end_day = test_start_day + test_length
        valid_start_day = test_start_day - valid_length
        
        
        """ Variation of the DATE model - only for research purposes (Measure the effect of smarter batch selection)
        # randomupDATE: Performance evaluation is done by DATE strategy, but newly added instances are random - not realistic)
        # noupDATE: DATE model does not accept new train data. (But the model anyway needs to be retrained with test data owing to the design choice of our XGB model) 
        These two strategies will be removed for software release. """
        
        if samp == 'noupDATE':
            data.update(data.df.loc[[]], data.df.loc[set(data.test.index)], test_start_day, test_end_day, valid_start_day)
        elif samp == 'randomupDATE':
            chosen = random.RandomSampling(data, args).query(num_samples)
            indices = [point + data.offset for point in chosen]         
            inspected_imports = data.df.loc[indices]
            uninspected_imports = data.df.loc[set(data.test.index)-set(inspected_imports.index)]            
            data.update(inspected_imports, uninspected_imports, test_start_day, test_end_day, valid_start_day)          
        else:
            data.update(inspected_imports, uninspected_imports, test_start_day, test_end_day, valid_start_day)
        
        
        del inspected_imports
        del uninspected_imports
        
        print("===========================================================================================")
        print()
