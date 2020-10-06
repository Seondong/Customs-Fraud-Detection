# Simulation Framework for Customs Selection 

Use your collected declarations data, fit the provided models, find the best selection strategy for your customs. 
This framework supports general import declarations. 



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


## Data Format
For your understanding, we upload the synthetic import declarations in the `data/` directory.
Users are expected to preprocess their import declarations in a similar format.
Currently, the framework supports single-item declarations that the target labels; illicitness of the item, revenue by inspection, are marked for each item.

|sgd.id|sgd.date  |importer.id| tariff.code| ... |cif.value|total.taxes|illicit|revenue|
|:----:|:--------:|:---------:|:----------:|:---:|--------:|----------:|:-----:|------:|
| SGD1 | 13-01-02 | IMP826164 |8703241128  | ... |2809     | 647       | 0     | 0     |
| SGD2 | 13-01-02 | IMP837219 |8703232926  | ... |266140   | 3262      | 0     | 0     |
| SGD3 | 13-01-02 | IMP117406 |8517180000  | ... |302275   | 5612      | 0     | 0     | 
| SGD4 | 13-01-02 | IMP435108 |8703222900  | ... |4160     | 514       | 0     | 0     |
| SGD5 | 13-01-02 | IMP717900 |8545200000  | ... |239549   | 397       | 1     | 980   |


## Available Selection Strategies:
* [Random](./query_strategies/random.py): Random selection, often used as a sub-strategy to find novel frauds by compensating the weakness of the selection model based on historical data. 
* [XGBoost](./query_strategies/xgb.py): Baseline exploitation strategy using both risky profiles of categorical variables and numeric variables as inputs. [[Reference]](https://xgboost.readthedocs.io/en/latest/python/python_api.html)
* [XBGoost + Logistic Regression](./query_strategies/xgb_lr.py): Two-stage strategy, selection is done by logistic regression results, and logistic regression is done by getting XGB leaf indices.
* [DATE](./query_strategies/DATE.py): Tree-aware dual attentive model for finding the most illicit and valuable imports at the same time; SOTA exploitation strategy. [[Reference]](https://github.com/Roytsai27/Dual-Attentive-Tree-aware-Embedding)
* [Tabnet](./query_strategies/tabnet.py): Tabnet trains an encoder-decoder structure to reconstruct masked tabular inputs. Selection is done with binary classification using a trained encoder. [[Reference]](https://github.com/dreamquark-ai/tabnet)
* [Diversity](./query_strategies/diversity.py): Variation of BADGE, this model uses the penultimate layer of the DATE model, then finds centroids by KMeans. In this way, diverse imports can be selected for the next inspection.
* [BADGE](./query_strategies/badge.py): BADGE model uses the gradient embedding of the base model (DATE) and find the most diverse imports by KMeans++. [[Reference]](https://github.com/JordanAsh/badge)
* [bATE](./query_strategies/bATE.py): Proposed model for better exploration. By following the BADGE model, we first use the embeddings of the base model, DATE. Our contribution is to amplify the embedding with extra uncertainty score, and predicted revenue. Finally, we find the most diverse imports by KMeans++.
* [Hybrid](./query_strategies/hybrid.py): Support mixing several strategies.
* [SSL-AE](./query_strategies/ssl_ae.py): Semi-supervised learning approach by optimizing reconstruction loss of all imports and binary cross-entropy of labeled imports


## Preliminary Results on Two Datasets
*[Data N](fig/ndata.png)
*[Data T](fig/tdata.png)


## Related Repositories
* DATE: Dual-Attentive Tree-aware Embedding for Customs Fraud Detection (KDD'2020) [[Link]](https://github.com/Roytsai27/Dual-Attentive-Tree-aware-Embedding)
* Machine Learning for Customs Fraud Detection [[Link]](https://github.com/YSCHOI-github/Customs_Fraud_Detection)


## Contribution
We welcome you to contribute to designing new selection strategies, automating feature engineering adaptive to different feature sets, donating anonymized import declarations dataset, and packaging software (PyPI).

