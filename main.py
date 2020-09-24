import logging
import preprocess_data
import generate_loader
import DATE_model
import argparse
import os
import csv
import pdb
import pickle
import warnings
import time 
import dataset
from collections import defaultdict
from datetime import timedelta
import datetime
import numpy as np
import torch
import torch.utils.data as Data
import pandas as pd
import torch
from model.AttTreeEmbedding import Attention, DATE
from ranger import Ranger
from utils import torch_threshold, metrics, metrics_active
from query_strategies import badge, badge_DATE, random_sampling, DATE_sampling, diversity, uncertainty, hybrid, xgb, xgb_lr, ssl_ae
warnings.filterwarnings("ignore")

def make_logger(curr_time, name=None):
    
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

    
def load_data(path):
    # load torch dataset 
    with open(path, "rb") as f:
        data = pickle.load(f)

    # get torch dataset
    train_dataset = data["train_dataset"]
    valid_dataset = data["valid_dataset"]
    test_dataset = data["test_dataset"]

    # create dataloader
    batch_size = 256
    train_loader = Data.DataLoader(
        dataset=train_dataset,     
        batch_size=batch_size,      
        shuffle=True,               
    )
    valid_loader = Data.DataLoader(
        dataset=valid_dataset,     
        batch_size=batch_size,      
        shuffle=False,               
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,     
        batch_size=batch_size,      
        shuffle=False,               
    )

    # parameters for model 
    leaf_num = data["leaf_num"]
    importer_size = data["importer_num"]
    item_size = data["item_size"]

    # global variables (xgb_testy: illicit or not)
    xgb_validy = valid_loader.dataset.tensors[-2].detach().numpy()
    xgb_testy = test_loader.dataset.tensors[-2].detach().numpy()
    revenue_valid = valid_loader.dataset.tensors[-1].detach().numpy()
    revenue_test = test_loader.dataset.tensors[-1].detach().numpy()

    return train_loader, valid_loader, test_loader, leaf_num, importer_size, item_size, xgb_validy, xgb_testy, revenue_valid, revenue_test


def evaluate_upDATE(chosen_rev,chosen_cls,xgb_testy,revenue_test):
    #get data
    logger.info("--------Evaluating the model---------")

    precisions, recalls, f1s, revenues = metrics_active(chosen_rev,chosen_cls,xgb_testy,revenue_test)
    best_score = f1s
    
    return precisions, recalls, f1s, revenues


if __name__ == '__main__':
    
    # Initiate directories
    curr_time = str(int(time.time()))
    
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists('./results/performances'):
        os.makedirs('./results/performances')
    if not os.path.exists('./results/query_indices'):
        os.makedirs('./results/query_indices')
    if not os.path.exists('./intermediary'):
        os.makedirs('./intermediary')    
    if not os.path.exists('./intermediary/saved_models'):
        os.makedirs('./intermediary/saved_models')   
    if not os.path.exists('./intermediary/logs'):
        os.makedirs('./intermediary/logs')
    if not os.path.exists('./intermediary/leaf_indices'):
        os.makedirs('./intermediary/leaf_indices')
    if not os.path.exists('./intermediary/processed_data'):
        os.makedirs('./intermediary/processed_data')
    if not os.path.exists('./intermediary/torch_data'):
        os.makedirs('./intermediary/torch_data')
    if not os.path.exists('./intermediary/xgb_models'):
        os.makedirs('./intermediary/xgb_models')
    if not os.path.exists('./uncertainty_models'):
        os.makedirs('./uncertainty_models')
    
    logger = make_logger(curr_time)
    
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="DATE", help="Name of the model")
    parser.add_argument('--epoch', type=int, default=5, help="Number of epochs")
    parser.add_argument('--dim', type=int, default=16, help="Hidden layer dimension")
    parser.add_argument('--lr', type=float, default=0.005, help="learning rate")
    parser.add_argument('--l2', type=float, default=0.01, help="l2 reg")
    parser.add_argument('--alpha', type=float, default=10, help="Regression loss weight")
    parser.add_argument('--beta', type=float, default=0.00, help="Adversarial loss weight")
    parser.add_argument('--head_num', type=int, default=4, help="Number of heads for self attention")
    parser.add_argument('--use_self', type=int, default=1, help="Whether to use self attention")
    parser.add_argument('--fusion', type=str, choices=["concat","attention"], default="concat", help="Fusion method for final embedding")
    parser.add_argument('--agg', type=str, choices=["sum","max","mean"], default="sum", help="Aggreate type for leaf embedding")
    parser.add_argument('--act', type=str, choices=["mish","relu"], default="relu", help="Activation function")
    parser.add_argument('--devices', type=str, default=['0','1','2','3'], help="list of gpu available")
    parser.add_argument('--device', type=str, default='0', help='select which device to run, choose gpu number in your devices or cpu') 
    parser.add_argument('--output', type=str, default="result"+"-"+curr_time, help="Name of output file")
    parser.add_argument('--sampling', type=str, default = 'badge_DATE', choices=['badge', 'badge_DATE', 'random', 'DATE', 'diversity', 'hybrid', 'xgb', 'xgb_lr', 'ssl_ae'], help='Sampling strategy')
    parser.add_argument('--initial_inspection_rate', type=int, default=100, help='Initial inspection rate in training data by percentile')
    parser.add_argument('--final_inspection_rate', type=int, default = 5, help='Percentage of test data need to query')
    parser.add_argument('--mode', type=str, default = 'finetune', choices = ['finetune', 'scratch'], help = 'finetune last model or train from scratch')
    parser.add_argument('--subsamplings', type=str, default = 'badge_DATE/DATE', help = 'available for hybrid sampling, the list of sub-sampling techniques seperated by /')
    parser.add_argument('--weights', type=str, default = '0.5/0.5', help = 'available for hybrid sampling, the list of weights for sub-sampling techniques seperated by /')
    parser.add_argument('--uncertainty', type=str, default = 'naive', choices = ['naive', 'self-supervised'], help = 'Uncertainty principle : ambiguity of illicitness or self-supervised manner prediction')
    parser.add_argument('--train_from', type=str, default = '20160105', help = 'Training period start from (YYYYMMDD)')
    parser.add_argument('--test_from', type=str, default = '20160112', help = 'Testing period start from (YYYYMMDD)')
    parser.add_argument('--test_length', type=int, default=3, help='Single testing period length (e.g., 7)')
    parser.add_argument('--valid_length', type=int, default=3, help='Validation period length (e.g., 7)')
    parser.add_argument('--data', type=str, default='real-n', choices = ['synthetic', 'real-n', 'real-m', 'real-t', 'real-k', 'real-c'], help = 'Dataset')
    parser.add_argument('--testnum', type=int, default=100, help='number of tests')
    parser.add_argument('--semi_supervised', type=int, default=0, help='Additionally using uninspected, unlabeled data (1=semi-supervised, 0=fully-supervised')
    parser.add_argument('--identifier', type=str, default=curr_time, help='identifier for each execution')
    
    # args
    args = parser.parse_args()
    epochs = args.epoch
    dim = args.dim
    lr = args.lr
    weight_decay = args.l2
    head_num = args.head_num
    act = args.act
    fusion = args.fusion
    alpha = args.alpha
    beta = args.beta
    use_self = args.use_self
    agg = args.agg
    samp = args.sampling
    ir_init = args.initial_inspection_rate
    perc = args.final_inspection_rate
    mode = args.mode
    unc_mode = args.uncertainty
    train_begin = args.train_from 
    test_begin = args.test_from
    test_length = args.test_length
    valid_length = args.valid_length
    chosen_data = args.data
    numTests = args.testnum
    semi_supervised = args.semi_supervised
    
    logger.info(args)
    
    
        
    if chosen_data == 'synthetic':
        data = dataset.Syntheticdata(path='./data/synthetic-imports-declarations.csv')
    elif chosen_data == 'real-n':
        data = dataset.Ndata(path='./data/ndata.csv')
    elif chosen_data == 'real-m':
        data = dataset.Mdata(path='./data/mdata.csv')
    elif chosen_data == 'real-t':
        data = dataset.Tdata(path='./data/tdata.csv')
    elif chosen_data == 'real-k':
        data = dataset.Kdata(path='./data/kdata.csv')  
    elif chosen_data == 'real-c':
        data = dataset.Cdata(path='./data/cdata.csv')  
    
        
    
    output_file =  "./results/performances/" + args.output + '-' + samp + '-' + str(perc) +".csv"
    
    with open(output_file, 'a') as ff:
        output_metric_name = ['runID', 'data', 'num_train','num_valid','num_test','num_select','num_total_newly_labeled','num_test_illicit','test_illicit_rate', 'upper_bound_precision', 'upper_bound_recall','upper_bound_rev', 'sampling', 'initial_inspection_rate', 'final_inspection_rate', 'mode', 'subsamplings', 'weights','unc_mode', 'train_start', 'valid_start', 'test_start', 'test_end', 'numWeek', 'precision', 'recall', 'revenue', 'norm-precision', 'norm-recall', 'norm-revenue']
        print(",".join(output_metric_name),file=ff)

    newly_labeled = None
    uncertainty_module = None
    path = None

    train_start_day = datetime.date(int(train_begin[:4]), int(train_begin[4:6]), int(train_begin[6:8]))
    test_start_day = datetime.date(int(test_begin[:4]), int(test_begin[4:6]), int(test_begin[6:8]))
    test_length = timedelta(days=test_length)    
    test_end_day = test_start_day + test_length
    valid_length = timedelta(days=valid_length)
    valid_start_day = test_start_day - valid_length
    initial_train_end_day = test_start_day
    
    
    # Customs selection simulation for long term
    for i in range(numTests):
        # make dataset                                    
        splitter = [train_start_day, initial_train_end_day, valid_start_day, test_start_day, test_end_day]               
        offset, train_labeled_data, valid_data, test_data = preprocess_data.split_data(data, splitter, curr_time, ir_init, semi_supervised, newly_labeled)
        logger.info('%s, %s', train_labeled_data.shape, test_data.shape)
        

        # get uncertainty from DATE for those needs it
        if unc_mode == 'self-supervised':
            if samp in ['badge_DATE', 'diversity', 'hybrid']:
                if uncertainty_module is None :
                    uncertainty_module = uncertainty.Uncertainty(train_labeled_data, './uncertainty_models/')
                    uncertainty_module.train()
                uncertainty_module.test_data = test_data 
        
        
        _, _, _, _, _, _, revenue_test, _, _, norm_revenue_test, _, _, _, _, _, xgb_testy = generate_loader.separate_train_test_data(curr_time)
        
        
        # Train XGB model only if the sampling strategy is dependent on XGB model.
        
        
        if not semi_supervised:
            if samp not in ['random']:
                generate_loader.prepare_input_for_DATE(curr_time)
                torchdata = load_data("./intermediary/torch_data/torch_data-"+curr_time+".pickle")
                train_loader, valid_loader, test_loader, leaf_num, importer_size, item_size, _, _, _, _ = torchdata

            # Train DATE model only if the sampling strategy is dependent on DATE model (except random and xgb).
            if samp not in ['random', 'xgb', 'xgb_lr']:
                # create / load model
                if mode == 'scratch' or i == 0:
                    date_model = DATE_model.VanillaDATE(torchdata, curr_time)
                else:
                    model = torch.load(path)
                    date_model = DATE_model.VanillaDATE(torchdata, curr_time, model.state_dict())
                # re-train
                date_model.train(args)
                overall_f1, auc, precisions, recalls, f1s, revenues, path = date_model.evaluate()

                logger.info("===========================================================================================")
                logger.info("""Metrics DATE:\nf1:%.4f auc:%.4f\nPr@5:%.4f Re@5:%.4fRev@5:%.4f""" \
                      % (overall_f1, auc,\
                         precisions[2],recalls[2],revenues[2]
                         ),
                         )

        if semi_supervised:
            if samp in ['ssl_ae']:
                generate_loader.prepare_input_for_SSL(curr_time)
                torchdata = load_data("./intermediary/torch_data/torch_ssl_data-"+curr_time+".pickle")
                train_loader_labeled, train_loader_unlabeled, valid_loader, test_loader = torchdata
                       

        # Selection stragies
        num_samples = int(len(test_data)*perc/100)
        
        
        def initialize_sampler(samp):
            if samp == 'random':
                sampler = random_sampling.RandomSampling(path, test_data, None, args)
            if samp == 'xgb':
                sampler = xgb.XGBSampling(path, test_data, None, args)
            if samp == 'xgb_lr':
                sampler = xgb_lr.XGBLRSampling(path, test_data, None, args)
            if samp == 'badge':
                sampler = badge.BadgeSampling(path, test_data, test_loader, args)
            if samp == 'DATE':
                sampler = DATE_sampling.DATESampling(path, test_data, test_loader, args)
            if samp == 'diversity':
                sampler = diversity.DiversitySampling(path, test_data, test_loader, uncertainty_module, args)
            if samp == 'badge_DATE':
                sampler = badge_DATE.DATEBadgeSampling(path, test_data, test_loader, uncertainty_module, args)
            if samp == 'ssl_ae':
                sampler = ssl_ae.SSLAutoencoderSampling(path, test_data, test_loader,args)
            return sampler
            
        if samp != 'hybrid':
            sampler = initialize_sampler(samp)
        
        if samp == 'hybrid':
            subsamplers = [initialize_sampler(samp) for samp in args.subsamplings.split("/")]
            weights = [float(weight) for weight in args.weights.split("/")]
            sampler = hybrid.HybridSampling(path, test_data, test_loader, args, subsamplers, weights)
        
        chosen = sampler.query(num_samples)
        logger.info("%s, %s, %s", len(set(chosen)), len(chosen), num_samples)
        assert len(set(chosen)) == num_samples
  
        # add new label:    
        indices = [point + offset for point in chosen]
        
        added_df = data.df.iloc[indices]
        
#         import pdb
#         pdb.set_trace()
    
        if newly_labeled is not None:
            newly_labeled = pd.concat([newly_labeled, added_df])
        else:
            newly_labeled = added_df
            
        logger.debug(added_df[:5])
        # tune the uncertainty
        
        if unc_mode == 'self-supervised' :
            if samp in ['badge_DATE', 'diversity', 'hybrid']:
                uncertainty_module.retrain(test_data.iloc[indices - offset])

        active_rev = added_df['revenue']
        active_rev = active_rev.transpose().to_numpy()

        active_cls = added_df['illicit']
        active_cls = active_cls.transpose().to_numpy()

        # evaluate
        active_precisions, active_recalls, active_f1s, active_revenues = evaluate_upDATE(active_rev,active_cls,xgb_testy,revenue_test)
        logger.info(f'Metrics Active DATE:\n Pr@{perc}:{round(active_precisions, 4)}, Re@{perc}:{round(active_recalls, 4)} Rev@{perc}:{round(active_revenues, 4)}') 
        
        with open(output_file, 'a') as ff:
            if samp == 'hybrid':
                subsamplings = args.subsamplings
                weights = args.weights
            else:
                subsamplings = '-'
                weights = '-'
                
            upper_bound_precision = min(100*np.mean(xgb_testy)/perc, 1)
            upper_bound_recall = min(perc/np.mean(xgb_testy)/100, 1)
            upper_bound_revenue = sum(sorted(revenue_test, reverse=True)[:len(chosen)]) / sum(revenue_test)
            norm_precision = active_precisions/upper_bound_precision
            norm_recall = active_recalls/upper_bound_recall
            norm_revenue = active_revenues/upper_bound_revenue
            
            
            output_metric = [curr_time, chosen_data, len(train_labeled_data), len(valid_data), len(test_data), len(chosen), len(newly_labeled), np.sum(xgb_testy), np.mean(xgb_testy), upper_bound_precision, upper_bound_recall, upper_bound_revenue, samp, ir_init, perc, mode, subsamplings, weights, unc_mode, train_start_day.strftime('%y-%m-%d'), valid_start_day.strftime('%y-%m-%d'), test_start_day.strftime('%y-%m-%d'), test_end_day.strftime('%y-%m-%d'), i+1, round(active_precisions,4), round(active_recalls,4), round(active_revenues,4), round(norm_precision,4), round(norm_recall,4), round(norm_revenue,4)]
                
                
            output_metric = list(map(str,output_metric))
            logger.debug(output_metric)
            print(",".join(output_metric),file=ff)
        
        
        output_file_indices =  "./results/query_indices/" + curr_time + '-' + samp + '-' + str(perc) + '-' + mode + "-week-" + str(i) + ".csv"
        
        with open(output_file_indices,"w") as queryFiles:
            wr = csv.writer(queryFiles, delimiter = ",")
            wr.writerow([i, test_start_day, test_end_day,indices])


        # Renew valid & test period 
        test_start_day = test_end_day
        test_end_day = test_start_day + test_length
        valid_start_day = test_start_day - valid_length

        logger.info("===========================================================================================")
