# Simulation Framework for Customs Selection 

A simulation framework for customs selection. Find the best selection strategy for future. This model supports general import declarations. 


## How to Install  
1. Setup your Python environment: e.g., Anaconda Python 3.7 [Guide](docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
```
$ source activate py37 
```

2. Clone the repository
```
$ https://github.com/john-mai-2605/WCO-project.git
```

3. Install requirements 
```
$ pip install -r requirements.txt
# Please install the Ranger optimizer by following its instruction.
```

4. Run the codes
```
$ export CUDA_VISIBLE_DEVICES=3 && python main.py --data real-m --semi_supervised 0 --sampling hybrid --subsamplings bATE/DATE --weights 0.1/0.9 --mode scratch --train_from 20130101 --test_from 20130701 --test_length 30 --valid_length 30 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay
```

## Available Selection Strategies:
* [Random](./query_strategies/random.py): Random selection
* [XGBoost](./query_strategies/xgb.py): Baseline explotation strategy using both risky profiles of categorical variables and numeric variables as inputs. [Reference](https://xgboost.readthedocs.io/en/latest/python/python_api.html)
* [XBGoost + Logistic Regression](./query_strategies/xgb_lr.py): Advanced version of XGB, logistic regression is done by XGB leave indices.
* [DATE](./query_strategies/DATE.py): Tree-aware dual attentive model for finding the most fraudy and valuable imports at the same time; SOTA explotation strategy. [Reference](https://github.com/Roytsai27/Dual-Attentive-Tree-aware-Embedding)
* [Tabnet](./query_strategies/tabnet.py): Using encoder structure of masked tabular inputs, to classify fraudness. [Reference](https://github.com/dreamquark-ai/tabnet)
* [Diversity](./query_strategies/diversity.py): Variation of BADGE, the model uses DATE embedding results, then select centroid by KMeans. In this way, diverse imports for next inspection are guaranteed.
* [BADGE](./query_strategies/badge.py) : BADGE model uses the embeddings of the base model (DATE) and find the most diverse imports by KMeans++. [Reference](https://github.com/JordanAsh/badge)
* [bATE](./query_strategies/bATE.py): Proposed model for better exploration. By following the BADGE model, we first uses the embeddings of the base model, DATE. Our contribution is to amplify the embedding with extra uncertainty score, and predicted revenue. Finally, we find the most diverse imports by KMeans++.
* [Hybrid](./query_strategies/hybrid.py): Support mixing several strategies.
* [SSL-AE](./query_strategies/ssl_ae.py): Semi-supervised learning by optimizing reconstruction loss of all imports and binary cross entropy of labeled imports

## Arguments
This is for explaining arguments for execution.

