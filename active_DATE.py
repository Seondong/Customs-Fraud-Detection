import preprocess_data
import generate_loader
import DATE_model
import argparse
import os
import pickle
import warnings
import time 
from collections import defaultdict
from datetime import timedelta
import datetime
from query_strategies import badge, badge_DATE, random_sampling, DATE_sampling, diversity, uncertainty
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import pandas as pd
import torch
from model.AttTreeEmbedding import Attention, DATE
from ranger import Ranger
from utils import torch_threshold, fgsm_attack, metrics, metrics_active

df = pd.read_csv('./data/synthetic-imports-declarations.csv', encoding = "ISO-8859-1")
df = df.dropna(subset=["illicit"])
df = df.sort_values("sgd.date")

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
    model_path = "./saved_models/%s%s.pkl" % (model_name,curr_time)

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
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    
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
    parser.add_argument('--device', type=str, choices=["cuda:0","cuda:1","cpu"], default="cuda:0", help="device name for training")
    parser.add_argument('--output', type=str, default="full.csv", help="Name of output file")
    parser.add_argument('--save', type=int, default=1, help="save model or not")
    parser.add_argument('--sampling', type=str, default = 'badge_DATE', choices=['badge', 'badge_DATE', 'random', 'DATE', 'diversity'], help='Sampling strategy')
    parser.add_argument('--percentage', type=int, default = 5, help='Percentage of test data need to query')
    parser.add_argument('--mode', type=str, default = 'finetune', choices = ['finetune', 'scratch'], help = 'finetune last model or train from scratch')
    # args
    args = parser.parse_args()
    epochs = args.epoch
    dim = args.dim
    lr = args.lr
    weight_decay = args.l2
    head_num = args.head_num
    save_model = args.save
    act = args.act
    fusion = args.fusion
    alpha = args.alpha
    beta = args.beta
    use_self = args.use_self
    agg = args.agg
    samp = args.sampling
    perc = args.percentage
    mode = args.mode
    print(args)

    numWeeks = 20
    newly_labeled = None
    start_day = datetime.date(2013, 4, 1)
    end_day = start_day + timedelta(days = 7)
    uncertainty_module = None

    for i in range(numWeeks):
        # make dataset
        splitter = ["13-01-01", "13-03-25", "13-03-25", "13-04-01", start_day.strftime('%y-%m-%d'), end_day.strftime('%y-%m-%d')]
        
        offset = preprocess_data.split_data(df, splitter, newly_labeled)
        print("offset %d" %offset)

        with open("./processed_data.pickle","rb") as f :
            processed_data = pickle.load(f)

        train_labeled_data = processed_data["raw"]["train"]
        test_data = processed_data["raw"]["test"]
        if uncertainty_module is None :
            uncertainty_module = uncertainty.Uncertainty(train_labeled_data)
            uncertainty_module.train()
        uncertainty_module.test_data = test_data
        
        generate_loader.loader()
        # load data
        data = load_data("./torch_data.pickle")
        revenue_upDATE = []
        train_loader, valid_loader, test_loader, leaf_num, importer_size, item_size, xgb_validy, xgb_testy, revenue_valid, revenue_test = data

        # create / load model
        if mode == 'scratch' or i == 0:
            date_model = DATE_model.VanillaDATE(data)
        else:
            model = torch.load(path)
            date_model = DATE_model.VanillaDATE(data, model.parameters())
        # re-train
        date_model.train(args)
        overall_f1, auc, precisions, recalls, f1s, revenues, path = date_model.evaluate(save_model)

        # save result
        output_file =  "./results/" + args.output
        print("Saving result...",output_file)
        with open(output_file, 'a') as ff:
            # print(args,file=ff)
            print()
            # print("""Metrics DATE:\nf1:%.4f auc:%.4f\nPr@1:%.4f Pr@2:%.4f Pr@5:%.4f Pr@10:%.4f\nRe@1:%.4f Re@2:%.4f Re@5:%.4f Re@10:%.4f\nRev@1:%.4f Rev@2:%.4f Rev@5:%.4f Rev@10:%.4f""" \
            #       % (overall_f1, auc,\
            #          precisions[0],precisions[1],precisions[2],precisions[3],\
            #          recalls[0],recalls[1],recalls[2],recalls[3],\
            #          revenues[0],revenues[1],revenues[2],revenues[3]
            #          ),
            #          )
            print("===========================================================================================")
            print("""Metrics DATE:\nf1:%.4f auc:%.4f\nPr@5:%.4f Re@5:%.4fRev@5:%.4f""" \
                  % (overall_f1, auc,\
                     precisions[2],recalls[2],revenues[2]
                     ),
                     )
            output_metric = [dim,overall_f1,auc] + precisions + recalls + revenues
            output_metric = list(map(str,output_metric))
            print(" ".join(output_metric),file=ff)

        # selection
        # testing top perc%
        num_samples = int(test_loader.dataset.tensors[-1].shape[0]*(perc/100))
        if samp == 'random':
            sampling = random_sampling.RandomSampling(path, test_loader, uncertainty_module, args)            
        elif samp == 'badge_DATE':
            sampling = badge_DATE.DATEBadgeSampling(path, test_loader, uncertainty_module, args)
        elif samp == 'badge':
            sampling = badge.BadgeSampling(path, test_loader, uncertainty_module, args)
        elif samp == 'DATE':
            sampling = DATE_sampling.DATESampling(path, test_loader, uncertainty_module, args)
        elif samp == 'diversity':
            sampling = diversity.DiversitySampling(path, test_loader, uncertainty_module, args)
        
        chosen = sampling.query(num_samples)
        # print(chosen)      
  
        # add new label:
        indices = [point + offset for point in chosen]
        added_df = df.iloc[indices]
        if newly_labeled is not None:
            newly_labeled = pd.concat([newly_labeled, added_df])
        else:
            newly_labeled = added_df                    
        # print(added_df[:5])
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

        # New week's starting day
        start_day = end_day
        end_day = start_day + timedelta(days = 7) 
        print("===========================================================================================")
