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
from scipy.stats import hmean
import torch
from model.AttTreeEmbedding import Attention, DATEModel
from utils import timer_func, evaluate_inspection, evaluate_inspection_multiclass
from query_strategies import uncertainty, random
warnings.filterwarnings("ignore")


class Simulator():
    """ Simulator class """

    def __init__(self):
        self.sim_start_time = str(round(time.time(),3))
        print('Experiment starts: ', self.sim_start_time)
        
        self.generate_paths()
        self.logger = make_logger(self.sim_start_time)
        
        # Parse argument
        parser = argparse.ArgumentParser()
        
        # Hyperparameters related to DATE
        parser.add_argument('--epoch', type=int, default=5, help="Number of epochs for DATE-related models")
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
        parser.add_argument('--risk_profile', type=str, choices=["topk","ratio"], default="topk", help="Risk profile criteria")
        
        # Hyperparameters related to customs selection
        parser.add_argument('--prefix', type=str, default='tkde-check', help="experiment name used as prefix for results file")
        parser.add_argument('--initial_masking', type=str, default="random", choices = ['random', 'importer', 'natural'], help="Masking some initial training data for simulating partially labeled scenario (for synthetic and m, n, t dataset)")
        parser.add_argument('--devices', type=str, default=['0','1','2','3'], help="list of gpu available")
        parser.add_argument('--device', type=str, default='0', help='select which device to run, choose gpu number in your devices or cpu') 
        parser.add_argument('--output', type=str, default="result"+"-"+self.sim_start_time, help="Name of output file")
        parser.add_argument('--sampling', type=str, default = 'xgb', choices=['random', 'risky', 'riskylogistic', 'riskyprod', 'riskyprec', 'riskyMAB', 'riskyMABsum', 'riskyDecayMAB', 'riskyDecayMABsum', 'AttentionAgg', 'AttentionAggRisky', 'xgb', 'xgb_lr', 'DATE', 'diversity', 'badge', 'bATE', 'upDATE', 'gATE', 'hybrid', 'adahybrid', 'tabnet', 'ssl_ae', 'deepSAD', 'multideepSAD', 'pot', 'pvalue', 'csi', 'rada'], help='Sampling strategy')
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
        parser.add_argument('--data', type=str, default='synthetic', choices = ['synthetic', 'synthetic-k', 'synthetic-k-partial', 'real-n', 'real-m', 'real-t', 'real-c', 'real-k'], help = 'Dataset')
        parser.add_argument('--numweeks', type=int, default=50, help='number of test weeks (week if test_length = 7)')
        parser.add_argument('--semi_supervised', type=int, default=0, help='Does the selection strategy use uninspected imports for training? (1 = Yes, Semi-supervised, 0 = No, fully-supervised)')
        parser.add_argument('--identifier', type=str, default=self.sim_start_time, help='identifier for each execution')
        parser.add_argument('--save', type=int, default=0, help='Save intermediary files (1=save, 0=not save)')

        # Hyperparameters for adahybrid.py and its childs:
        parser.add_argument('--ada_algo', type=str, choices = ['ucb', 'exp3', 'exp3s'], default='-', help="algorithm for adahybrid")
        parser.add_argument('--ada_discount', type=str, choices = ['window', 'decay'], default='-', help="algorithm for adahybrid")
        parser.add_argument('--ada_lr', type=float, default=0.8, help="learning rate for adahybrid")
        parser.add_argument('--ada_decay', type=float, default=1, help="decay factor for adahybrid, 1 for no decay")
        parser.add_argument('--ada_epsilon', type=float, default=0, help="degree of randomness for adahybrid")
        parser.add_argument('--num_arms', type=int, default=21, help="number of arms for adahybrid")

        # Hyperparameters for radahybrid.py:
        parser.add_argument('--drift', type=str, default='pot', choices = ['pot', 'pvalue', 'csi'], help="algorithms for measuring concept drift")
        parser.add_argument('--mixing', type=str, default='multiply', choices = ['multiply', 'reinit', 'balance'], help="method of mixing concept drift with regulated adahybrid")

        # Arguments
        args = parser.parse_args()
        self.args = args        
        self.logger.info(self.args)

        self.hybrid_strategies = ['hybrid', 'adahybrid', 'pot', 'pvalue', 'csi', 'rada']
        self.uncertainty_module = None 
        
        # Initial dataset split
        self.train_start_day = datetime.date(int(args.train_from[:4]), int(args.train_from[4:6]), int(args.train_from[6:8]))
        self.test_start_day = datetime.date(int(args.test_from[:4]), int(args.test_from[4:6]), int(args.test_from[6:8]))
        self.test_length = timedelta(days=args.test_length)    
        self.test_end_day = self.test_start_day + self.test_length
        self.valid_length = timedelta(days=args.valid_length)
        self.valid_start_day = self.test_start_day - self.valid_length

        self.confirmed_inspection_plan = inspection_plan(args.initial_inspection_rate, args.final_inspection_rate, args.numweeks, args.inspection_plan)
        self.logger.info('Inspection rate for testing periods: %s', self.confirmed_inspection_plan)


    def generate_paths(self):
        """ Generate required directories """
        pathlib.Path('./results').mkdir(parents=True, exist_ok=True) 
        pathlib.Path('./results/performances').mkdir(parents=True, exist_ok=True)
        pathlib.Path('./results/ratios').mkdir(parents=True, exist_ok=True)    
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


    def prepare_data(self):
        """ Prepare data to simulate """

        # Load datasets 
        if self.args.data == 'synthetic':
            self.data = dataset.Syntheticdata(path='./data/synthetic-imports-declarations.csv')
        elif self.args.data == 'synthetic-k':
            self.data = dataset.SyntheticKdata(path='./data/df_syn_ano_0429_merge.csv')  # fully labeled
        elif self.args.data == 'synthetic-k-partial':
            self.data = dataset.SyntheticKdata(path='./data/df_syn_ano_0429_merge_partially_labeled.csv')   # partially labeled
            self.args.initial_masking = 'natural'   # since this data is given as partially labeled, it does not need extra label masking.
            initial_masking = 'natural'   
        elif self.args.data == 'real-k':
            self.data = dataset.Kdata(path='./data/kdata.csv')
            self.args.initial_masking = 'natural'
            initial_masking = 'natural'
        elif self.args.data == 'real-n':
            self.data = dataset.Ndata(path='./data/ndata.csv')
        elif self.args.data == 'real-m':
            self.data = dataset.Mdata(path='./data/mdata.csv')
        elif self.args.data == 'real-t':
            self.data = dataset.Tdata(path='./data/tdata.csv')
        elif self.args.data == 'real-c':
            self.data = dataset.Cdata(path='./data/cdata.csv')
        
        self.data.split(self.train_start_day, self.valid_start_day, self.test_start_day, self.test_end_day, self.valid_length, self.test_length, self.args)
        

    def prepare_output_files(self):
        """ Prepare output files """

        samp = self.args.sampling
        subsamps = self.args.subsamplings.replace('/','+')
        if self.args.sampling not in self.hybrid_strategies:
            subsamps = 'single'

        # Saving simulation results: Output file will be saved under ./results/performances/ directory
        self.output_file_desc = self.args.prefix + '-' + self.args.output + '-' + self.args.data + '-' + samp + '-' + subsamps + '-' + str(self.args.final_inspection_rate) 
        self.output_file = "./results/performances/" + self.output_file_desc + ".csv"
        with open(self.output_file, 'a') as ff:
            output_metric_name = ['runID', 'data', 'num_train','num_valid','num_test','num_select','num_inspected','num_uninspected','num_test_illicit','test_illicit_rate', 'upper_bound_precision', 'upper_bound_recall','upper_bound_rev', 'sampling', 'concept_drift', 'mixing',  'ada_lr', 'ada_decay', 'ada_epsilon', 'initial_inspection_rate', 'current_inspection_rate', 'final_inspection_rate', 'inspection_plan', 'mode', 'subsamplings', 'initial_weights', 'current_weights', 'unc_mode', 'train_start', 'valid_start', 'test_start', 'test_end', 'numWeek', 'precision', 'recall', 'revenue', 'avg_revenue', 'norm-precision', 'norm-recall', 'norm-revenue']
            print(",".join(output_metric_name),file=ff)
        
        # Additional output files for adaptive strategies
        if samp == 'adahybrid':
            self.weight_file =  "./results/ratios/" + self.args.prefix + '-' + self.args.output + '-' + samp + '-' + subsamps + '-' + str(self.args.final_inspection_rate) + ".csv"
            with open(self.weight_file, 'a') as ff:
                output_weight_name = ['runID', 'data', 'sampling', 'subsamplings', 'numWeek', 'norm-precision', 'norm-recall', 'norm-revenue', 'lr'] + [f'{i/(self.args.num_arms-1)} explore rate' for i in range(self.args.num_arms)] + ['chosen_rate', 'chosen_arm']
                print(",".join(output_weight_name), file=ff)

        # Additional output files for adaptive strategies
        if samp == 'rada':
            self.weight_file =  "./results/ratios/" + self.args.prefix + '-' + self.args.output + '-' + samp + '-' + subsamps + '-' + str(self.args.final_inspection_rate) + ".csv"
            with open(self.weight_file, 'a') as ff:
                output_weight_name = ['runID', 'data', 'sampling', 'subsamplings', 'numWeek', 'norm-precision', 'norm-recall', 'norm-revenue', 'lr', 'drift', 'mixing', 'drift_weight'] + [f'{i/(self.args.num_arms-1)} explore rate' for i in range(self.args.num_arms)] + ['chosen_rate', 'chosen_arm']
                print(",".join(output_weight_name), file=ff)


    def evaluate_results(self):
        """ Evaluation - calculate necessary metrics """

        # Evaluation
        rev = self.inspected_imports['revenue']
        rev = rev.transpose().to_numpy()

        cls = self.inspected_imports['illicit']
        cls = cls.transpose().to_numpy()

        # Added to handle semi-supervised inputs
        cls_notna = cls[~np.isnan(cls)]
        rev_notna = rev[~np.isnan(rev)]
        illicit_test_notna = self.data.test_cls_label[~np.isnan(self.data.test_cls_label)]
        revenue_test_notna = self.data.test_reg_label[~np.isnan(self.data.test_reg_label)]

        if self.args.data in ['synthetic-k', 'synthetic-k-partial', 'real-k']:
            metric_dict = evaluate_inspection_multiclass(self.inspected_imports, self.data.test, self.data.class_labels)
            self.logger.info(f'Performance:\n specific: MacroF1@{self.current_inspection_rate}:{round(metric_dict["specific_result"]["macrof1"], 4)}\n broad: MacroF1@{self.current_inspection_rate}:{round(metric_dict["broad_result"]["macrof1"], 4)}')

        self.precision, self.recall, self.f1, self.revenue_avg, self.revenue_recall = evaluate_inspection(rev_notna, cls_notna, illicit_test_notna, revenue_test_notna)
        self.logger.info(f'Performance:\n Pr@{self.current_inspection_rate}:{round(self.precision, 4)}, Re@{self.current_inspection_rate}:{round(self.recall, 4)} Rev@{self.current_inspection_rate}:{round(self.revenue_recall, 4)}') 

        self.upper_bound_precision = min(np.sum(illicit_test_notna)/len(self.inspected_indices), 1)
        self.upper_bound_recall = min(len(self.chosen)/np.sum(illicit_test_notna), 1)
        self.upper_bound_revenue = min(sum(sorted(revenue_test_notna, reverse=True)[:len(self.chosen)]) / np.sum(revenue_test_notna), 1)
        
        self.norm_precision = self.precision/self.upper_bound_precision
        self.norm_recall = self.recall/self.upper_bound_recall
        self.norm_revenue = self.revenue_recall/self.upper_bound_revenue


    def save_results(self):
        """ Save evaluation results"""
        samp = self.args.sampling

        if samp in self.hybrid_strategies:
            subsamplings = self.args.subsamplings
            initial_weights = [float(weight) for weight in self.args.weights.split("/")]
            final_weights = initial_weights
        else:
            subsamplings = '-'
            initial_weights = '-'
            final_weights = '-'

        if samp == 'rada':
            drift = self.args.drift
            mixing = self.args.mixing
        else:
            drift = '-'
            mixing = '-'
        
        if samp in ['adahybrid', 'rada']:
            ada_decay = self.args.ada_decay
            ada_epsilon = self.args.ada_epsilon
            ada_lr = self.args.ada_lr
        else:
            ada_decay = '-'
            ada_epsilon = '-'
            ada_lr = '-'

        with open(self.output_file, 'a') as ff:
            if samp in self.hybrid_strategies:
                initial_weights_str = '/'.join([str(weight) for weight in initial_weights])
                final_weights_str = '/'.join([str(weight) for weight in final_weights])

            else:
                initial_weights_str = '-'
                final_weights_str = '-'
            
            output_metric = [self.sim_start_time, self.args.data, len(self.data.train_lab), len(self.data.valid_lab), len(self.data.test), len(self.chosen), len(self.inspected_imports), len(self.uninspected_imports), np.sum(self.data.test_cls_label), np.mean(self.data.test_cls_label), self.upper_bound_precision, self.upper_bound_recall, self.upper_bound_revenue, samp, drift, mixing, ada_lr, ada_decay, ada_epsilon, self.args.initial_inspection_rate, self.current_inspection_rate, self.args.final_inspection_rate, self.args.inspection_plan, self.args.mode, subsamplings, initial_weights_str, final_weights_str, self.args.uncertainty, self.train_start_day.strftime('%y-%m-%d'), self.valid_start_day.strftime('%y-%m-%d'), self.test_start_day.strftime('%y-%m-%d'), self.test_end_day.strftime('%y-%m-%d'), self.data.episode+1, round(self.precision,4), round(self.recall,4), round(self.revenue_recall,4), round(self.revenue_avg,4), round(self.norm_precision,4), round(self.norm_recall,4), round(self.norm_revenue,4)]
                
            output_metric = list(map(str,output_metric))
            self.logger.debug(output_metric)
            print(",".join(output_metric),file=ff)

        if samp in ['adahybrid', 'rada']:
            with open(self.weight_file, 'a') as ff:
                subsamplings = self.args.subsamplings
                weights = '/'.join([str(weight) for weight in final_weights])
                
                output_metric = [self.sim_start_time, self.args.data, samp, subsamplings, self.data.episode+1, round(self.norm_precision,4), round(self.norm_recall,4), round(self.norm_revenue,4), ada_lr] + list(self.sampler.weight_sampler.p) + [self.sampler.weight_sampler.value, self.sampler.weight_sampler.arm]
                if samp == 'rada':
                    output_metric += [drift, mixing, self.sampler.drift_detector.dms_weight]
                    
                output_metric = list(map(str,output_metric))
                self.logger.debug(output_metric)
                print(",".join(output_metric),file=ff)
        
        if self.args.save == 1:
            # generate a folder
            self.output_indices_folder =  "./results/query_indices/" + self.output_file_desc+"/"
            pathlib.Path(self.output_indices_folder).mkdir(parents=True, exist_ok=True)
            self.output_file_indices = self.output_indices_folder+"results-ep"+str(self.data.episode)+"-"+str(self.test_start_day)+'-'+str(self.test_end_day)+".csv"                
                
            with open(self.output_file_indices, "w", newline='') as queryFiles:
                wr = csv.writer(queryFiles, delimiter = ",")
                wr.writerow(['Experiment ID', self.sim_start_time])
                wr.writerow(['Dataset', self.args.data])
                wr.writerow(['Episode', self.data.episode])
                wr.writerow(['Test_start_day', self.test_start_day])
                wr.writerow(['Test_end_day', self.test_end_day])
                
                if samp in self.hybrid_strategies:
                    tmpIdx = 0
                    for subsampler, num in zip(self.args.subsamplings.split('/'), self.sampler.ks):
                        row = [subsampler]
                        row.extend(self.inspected_indices[tmpIdx:tmpIdx+num])
                        wr.writerow(row)
                        tmpIdx += num
                else:
                    row = [samp]
                    row.extend(self.inspected_indices)
                    wr.writerow(row)


    def simulate(self):
        """ Main custom selection simulation part """

        # Initialize a sampler (We put it outside the week loop since we do not change sampler every week)
        # NOTE: If you put this inside the week loop, new sampler will be initialized every week, which means that parameters in the sampler are also initialized)    
        self.sampler = initialize_sampler(self.args.sampling, self.args)
        samp = self.args.sampling      
            
        # Customs selection simulation for long term (if test_length = 7 days, simulate for numweeks)
        for i in range(self.args.numweeks):
            
            # Terminating condition
            if self.test_start_day.strftime('%y-%m-%d') > max(self.data.df["sgd.date"]):
                self.logger.info('Simulation period is over.')
                self.logger.info('Terminating ...')
                sys.exit()

            # Feature engineering for train, valid, test data
            self.data.episode = i
            self.current_inspection_rate = self.confirmed_inspection_plan[i]  
            self.logger.info(f'Test episode: #{i}, Current inspection rate: {self.current_inspection_rate}')
            
            if samp not in ['random']: 
                self.data.featureEngineering()
            else:
                self.data.offset = self.data.test.index[0]
            
            # Initialize uncertainty module for some cases
            if self.args.uncertainty == 'self-supervised':
                if samp in ['bATE', 'diversity', 'hybrid', 'upDATE', 'gATE', 'adahybrid', 'pot', 'pvalue', 'csi', 'rada']:
                    if self.uncertainty_module is None :
                        self.uncertainty_module = uncertainty.Uncertainty(self.data.train_lab, './uncertainty_models/')
                        self.uncertainty_module.train()
                    self.uncertainty_module.test_data = self.data.test 
            
            # Number of items to inspect for this episode
            num_samples = int(len(self.data.test)*self.current_inspection_rate/100)
            
            # Retrieve subsampler weights from the previous week, for hybrid models
            if samp in self.hybrid_strategies:
                try:
                    final_weights = self.sampler.get_weights()
                except NameError:
                    pass  # use the previously defined final_weights (= initial_weights)
            
            # set uncertainty module
            self.sampler.set_uncertainty_module(self.uncertainty_module)
            
            # set previous weeks' weights, for hybrid models
            if samp in self.hybrid_strategies:
                self.sampler.set_weights(final_weights)
            
            # set data to sampler
            self.sampler.set_data(self.data)
            
            # query selection
            try:
                # import pdb
                # pdb.set_trace()
                self.chosen = self.sampler.query(num_samples)  
            except:
                import traceback
                traceback.print_exc()

            self.logger.info("--------Evaluating selection results---------")   
            self.logger.info("# of queried item: %s, # of samples to be queried: %s", len(self.chosen), num_samples)
            try:
                assert len(set(self.chosen)) == num_samples
            except AssertionError:
                import traceback
                traceback.print_exc()        
    
            # Indices of sampled imports (Considered as high-risky by model) -> This will be inspected thus annotated.    
            self.inspected_indices = [point + self.data.offset for point in self.chosen]
            
            # Originally, chosen trade should be annotated.
            # Compatible with simulating on synthetic-k-partial dataset. We need this procedure to evaluate the selection strategy on given partially-labeled datasets. 
            self.inspected_indices = self.data.df['illicit'][self.inspected_indices].notnull().loc[lambda x: x==True].index.values
            
            self.inspected_imports = self.data.df.iloc[self.inspected_indices]
            self.uninspected_imports = self.data.df.loc[set(self.data.test.index)-set(self.inspected_imports.index)]
            self.uninspected_imports['illicit'] = float('nan')
            self.uninspected_imports['revenue'] = float('nan')
        
            self.logger.debug(self.inspected_imports[:5])
            
            # tune the uncertainty
            if self.args.uncertainty == 'self-supervised' and samp in ['bATE', 'diversity', 'hybrid', 'gATE', 'adahybrid', 'pot', 'pvalue', 'csi', 'rada']:
                self.uncertainty_module.retrain(self.data.test.iloc[self.inspected_indices - self.data.offset])
            
            # Evaluate the inspected results
            self.evaluate_results()

            # Save the evaluation results into csv files
            self.save_results()

            # Review needed: Check if the weights are updated as desired.
            if samp in ['adahybrid', 'rada']:
                self.sampler.update_subsampler_weights(self.norm_precision)

            # Renew valid & test period & dataset
            if i == self.args.numweeks - 1:
                self.logger.info('Simulation period is over.')
                self.logger.info('Terminating ...')
                sys.exit()
                
            self.test_start_day = self.test_end_day
            self.test_end_day = self.test_start_day + self.test_length
            self.valid_start_day = self.test_start_day - self.valid_length
            
            self.data.update(self.inspected_imports, self.uninspected_imports, self.test_start_day, self.test_end_day, self.valid_start_day)
            
            
            del self.inspected_imports
            del self.uninspected_imports
            
            print("===========================================================================================")
            print()


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


def inspection_plan(rate_init, rate_final, numweeks, option):
    """ Inspection plan for next n weeks, add reasonable decaying strategy if necessary """
    if option == 'direct_decay':
        return np.linspace(rate_final, rate_final, numweeks)
    
    if option == 'linear_decay':
        return np.linspace(rate_init, rate_final, numweeks)
    
    if option == 'fast_linear_decay':
        first_half = np.linspace(rate_init, rate_final, 10)
        second_half = np.linspace(rate_final, rate_final, numweeks - len(first_half))
        return np.concatenate((first_half, second_half))


# Selection stragies
def initialize_sampler(samp, args):
    """Initialize selection strategies"""
    if samp == 'random':
        from query_strategies import random;
        sampler = random.RandomSampling(args)
    elif samp == 'risky':
        from query_strategies import risky;
        sampler = risky.RiskProfileSampling(args)
    elif samp == 'riskylogistic':
        from query_strategies import risky; 
        sampler = risky.RiskProfileLogisticSampling(args)
    elif samp == 'riskyprod':
        from query_strategies import risky; 
        sampler = risky.RiskProfileProdSampling(args)
    elif samp == 'riskyprec':
        from query_strategies import risky; 
        sampler = risky.RiskProfilePrecisionSampling(args)
    elif samp == 'riskyMAB':
        from query_strategies import risky; 
        sampler = risky.RiskProfileMABSampling(args)
    elif samp == 'riskyMABsum':
        from query_strategies import risky; 
        sampler = risky.RiskProfileMABSumSampling(args)
    elif samp == 'riskyDecayMAB':
        from query_strategies import risky; 
        sampler = risky.RiskProfileDiscountMABSampling(args)
    elif samp == 'riskyDecayMABsum':
        from query_strategies import risky; 
        sampler = risky.RiskProfileDiscountMABSumSampling(args)
    elif samp == 'AttentionAgg':
        from query_strategies import AttentionAggregate;
        sampler = AttentionAggregate.AttentionSampling(args)
    elif samp == 'AttentionAggRisky':
        from query_strategies import AttentionAggregate;
        sampler = AttentionAggregate.AttentionPlusRiskSampling(args)
    elif samp == 'xgb':
        from query_strategies import xgb;
        sampler = xgb.XGBSampling(args)
    elif samp == 'xgb_lr':
        from query_strategies import xgb_lr;
        sampler = xgb_lr.XGBLRSampling(args)
    elif samp == 'badge':
        from query_strategies import badge;
        sampler = badge.BadgeSampling(args)
    elif samp == 'DATE':
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
    elif samp == 'csi':
        from query_strategies import csi;
        sampler = csi.CSISampling(args)        
    elif samp == 'adahybrid':
        from query_strategies import adahybrid;
        sampler = adahybrid.AdaHybridSampling(args)
    elif samp == 'pvalue':
        from query_strategies import p_value;
        sampler = p_value.pvalueSampling(args)
    elif samp == 'rada':
        from query_strategies import radahybrid;
        sampler = radahybrid.RegulatedAdaHybridSampling(args)
    elif samp == 'multiclass':
        from query_strategies import multiclass;
        sampler = multiclass.MulticlassSampling(args)
    else:
        sampler = None
        print('Make sure the sampling strategy is listed in the argument --sampling')
    return sampler

def main():
    sim = Simulator()
    sim.prepare_data()
    sim.prepare_output_files()
    sim.simulate()


if __name__ == '__main__':
    main()
    

    