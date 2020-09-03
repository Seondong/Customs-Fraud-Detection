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
from collections import defaultdict
from datetime import timedelta
import datetime
from query_strategies import badge, badge_DATE, random_sampling, DATE_sampling, diversity, uncertainty, hybrid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import pandas as pd
import torch
from model.AttTreeEmbedding import Attention, DATE
from ranger import Ranger
from utils import torch_threshold, metrics, metrics_active
warnings.filterwarnings("ignore")


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

    # model information
    curr_time = str(time.time())
    model_name = "DATE"
    model_path = "./intermediary/python/%s%s.pkl" % (model_name,curr_time)

    return train_loader, valid_loader, test_loader, leaf_num, importer_size, item_size, xgb_validy, xgb_testy, revenue_valid, revenue_test


def evaluate_upDATE(chosen_rev,chosen_cls,xgb_testy,revenue_test):
    #get data
    print()
    print("--------Evaluating Active DATE model---------")

    precisions, recalls, f1s, revenues = metrics_active(chosen_rev,chosen_cls,xgb_testy,revenue_test)
    best_score = f1s
    
    return precisions, recalls, f1s, revenues


if __name__ == '__main__':
    
    # Parse argument
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
    if not os.path.exists('./uncertainty_models'):
        os.makedirs('./uncertainty_models')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', 
                        type=str, 
                        default="DATE", 
                        help="Name of model",
                        )
    parser.add_argument('--epoch', 
                        type=int, 
                        default=5, 
                        help="Number of epochs",
                        )
    parser.add_argument('--dim', 
                        type=int, 
                        default=16, 
                        help="Hidden layer dimension",
                        )
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.005, 
                        help="learning rate",
                        )
    parser.add_argument('--l2',
                        type=float,
                        default=0.01,
                        help="l2 reg",
                        )
    parser.add_argument('--alpha',
                        type=float,
                        default=10,
                        help="Regression loss weight",
                        )
    parser.add_argument('--beta', type=float, default=0.00, help="Adversarial loss weight")
    parser.add_argument('--head_num', type=int, default=4, help="Number of heads for self attention")
    parser.add_argument('--use_self', type=int, default=1, help="Whether to use self attention")
    parser.add_argument('--fusion', type=str, choices=["concat","attention"], default="concat", help="Fusion method for final embedding")
    parser.add_argument('--agg', type=str, choices=["sum","max","mean"], default="sum", help="Aggreate type for leaf embedding")
    parser.add_argument('--act', type=str, choices=["mish","relu"], default="relu", help="Activation function")
    parser.add_argument('--device', type=str, choices=["cuda:0","cuda:1","cuda:2","cuda:3","cpu"], default="cuda:0", help="device name for training")
    parser.add_argument('--output', type=str, default="result"+"-"+curr_time, help="Name of output file")
    parser.add_argument('--sampling', type=str, default = 'badge_DATE', choices=['badge', 'badge_DATE', 'random', 'DATE', 'diversity', 'hybrid'], help='Sampling strategy')
    parser.add_argument('--percentage', type=int, default = 5, help='Percentage of test data need to query')
    parser.add_argument('--mode', type=str, default = 'finetune', choices = ['finetune', 'scratch'], help = 'finetune last model or train from scratch')
    parser.add_argument('--subsamplings', type=str, default = 'badge_DATE/DATE', help = 'available for hybrid sampling, the list of sub-sampling techniques seperated by /')
    parser.add_argument('--weights', type=str, default = '0.5/0.5', help = 'available for hybrid sampling, the list of weights for sub-sampling techniques seperated by /')
    parser.add_argument('--uncertainty', type=str, default = 'naive', choices = ['naive', 'self-supervised'], help = 'Uncertainty principle : ambiguity of illicitness or self-supervised manner prediction')
    
    parser.add_argument('--train_from', type=str, default = '20160105', help = 'Training period start from (YYYYMMDD)')
    parser.add_argument('--test_from', type=str, default = '20160112', help = 'Testing period start from (YYYYMMDD)')
    parser.add_argument('--test_length', type=int, default=3, help='Single testing period length (e.g., 7)')
    parser.add_argument('--valid_length', type=int, default=3, help='Validation period length (e.g., 7)')
    
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
    perc = args.percentage
    mode = args.mode
    unc_mode = args.uncertainty
    train_begin = args.train_from 
    test_begin = args.test_from
    test_length = args.test_length
    valid_length = args.valid_length
    
    print(args)
    
    df = pd.read_csv('./data/ndata.csv', encoding = "ISO-8859-1")
    df = df.dropna(subset=["illicit"])
    df = df.sort_values("sgd.date")
    df = df.reset_index(drop=True)
    
    output_file =  "./results/performances/" + args.output + '-' + samp + '-' + str(perc) +".csv"
    with open(output_file, 'a') as ff:
        output_metric_name = ['num_train','num_test','num_select','num_total_newly_labeled','num_test_illicit','test_illicit_rate','upper_bound_recall','upper_bound_rev', 'sampling', 'percentage', 'mode', 'subsamplings', 'weights','unc_mode', 'train_start', 'valid_start', 'test_start', 'test_end', 'numWeek', 'precision', 'recall', 'revenue']
        print(" ".join(output_metric_name),file=ff)

    numTests = 100
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
        offset, train_labeled_data, valid_data, test_data = preprocess_data.split_data(df, splitter, curr_time, newly_labeled)
        print(train_labeled_data.shape, test_data.shape)

        # get uncertainty from DATE for those needs it
        if unc_mode == 'self-supervised' :
            if samp in ['badge_DATE', 'diversity', 'hybrid']:
                if uncertainty_module is None :
                    uncertainty_module = uncertainty.Uncertainty(train_labeled_data, './uncertainty_models/')
                    uncertainty_module.train()
                uncertainty_module.test_data = test_data
        
        generate_loader.loader(curr_time)
        
        data = load_data("./intermediary/torch_data-"+curr_time+".pickle")
        revenue_upDATE = []
        train_loader, valid_loader, test_loader, leaf_num, importer_size, item_size, xgb_validy, xgb_testy, revenue_valid, revenue_test = data
        if samp != 'random':
            # create / load model
            if mode == 'scratch' or i == 0:
                date_model = DATE_model.VanillaDATE(data, curr_time)
            else:
                model = torch.load(path)
                date_model = DATE_model.VanillaDATE(data, curr_time, model.state_dict())
            # re-train
            date_model.train(args)
            overall_f1, auc, precisions, recalls, f1s, revenues, path = date_model.evaluate()

            print("===========================================================================================")
            print("""Metrics DATE:\nf1:%.4f auc:%.4f\nPr@5:%.4f Re@5:%.4fRev@5:%.4f""" \
                  % (overall_f1, auc,\
                     precisions[2],recalls[2],revenues[2]
                     ),
                     )

        # selection
        # testing top perc%
        num_samples = int(test_loader.dataset.tensors[-1].shape[0]*(perc/100))
        samplers = {
                'random': random_sampling.RandomSampling(path, test_loader, args),            
                'badge_DATE': badge_DATE.DATEBadgeSampling(path, test_loader, uncertainty_module, args),
                'badge': badge.BadgeSampling(path, test_loader, args),
                'DATE': DATE_sampling.DATESampling(path, test_loader, args),
                'diversity': diversity.DiversitySampling(path, test_loader, uncertainty_module, args)}
        if samp == 'hybrid':
        	subsamps = [samplers[subsamp] for subsamp in args.subsamplings.split("/")]
        	weights = [float(weight) for weight in args.weights.split("/")]
        	sampling = hybrid.HybridSampling(path, test_loader, args, subsamps, weights)
        else:
        	sampling = samplers[samp]
        chosen = sampling.query(num_samples)
        print(len(set(chosen)), len(chosen), num_samples)
        assert len(set(chosen)) == num_samples
  
        # add new label:    
        indices = [point + offset for point in chosen]
        added_df = df.iloc[indices]
        
#         import pdb
#         pdb.set_trace()
    
        if newly_labeled is not None:
            newly_labeled = pd.concat([newly_labeled, added_df])
        else:
            newly_labeled = added_df                    
        print(added_df[:5])
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
        print("""Metrics Active DATE:\n Pr@5:%.4f Re@5:%.4f Rev@5:%.4f""" \
                  % (
                     active_precisions, active_recalls, active_revenues
                     ),
                     ) 
        
        with open(output_file, 'a') as ff:
            if mode == 'hybrid':
                subsamplings = args.subsamplings
                weights = arg.weights
            else:
                subsamplings = '-'
                weights = '-'
            output_metric = [len(train_labeled_data), len(xgb_testy), len(chosen), len(newly_labeled), np.sum(xgb_testy), np.mean(xgb_testy), min(perc/np.mean(xgb_testy)/100, 1), sum(sorted(revenue_test, reverse=True)[:len(chosen)]) / sum(revenue_test), samp, perc, mode, subsamplings, weights, unc_mode, train_start_day.strftime('%y-%m-%d'), valid_start_day.strftime('%y-%m-%d'), test_start_day.strftime('%y-%m-%d'), test_end_day.strftime('%y-%m-%d'), i+1, round(active_precisions,4), round(active_recalls,4), round(active_revenues,4)]
            output_metric = list(map(str,output_metric))
            print(output_metric)
            print(" ".join(output_metric),file=ff)
        
        
        output_file_indices =  "./results/query_indices/" + curr_time + '-' + samp + '-' + str(perc) + '-' + mode + "-week-" + str(i) + ".csv"
        
        with open(output_file_indices,"w") as queryFiles:
            wr = csv.writer(queryFiles,delimiter = ",")
            wr.writerow([i, test_start_day, test_end_day,indices])


        # Renew valid & test period 
        test_start_day = test_end_day
        test_end_day = test_start_day + test_length
        valid_start_day = test_start_day - valid_length
        print("===========================================================================================")
