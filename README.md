# Simulation Framework for Customs Selection 

Use your collected declarations data, fit the provided models, find the best selection strategy for your customs. 
This framework supports general import declarations. 


## How to Use  

1. Setup your Python environment: e.g., Anaconda Python 3.8 [[Guide]](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
```
$ conda activate py38 
```

2. Clone the repository:
```
$ git clone https://github.com/Seondong/Customs-Fraud-Detection.git
```

3. Install requirements 
```
$ pip install -r requirements.txt
```

4. Run the codes: Refer to main.py for hyperparameters, .sh files in `./bash` directory will give you some ideas how to run codes effectively. 
```
$ python main.py --data synthetic --train_from 20130101 --test_from 20130115 --valid_length 7 --test_length 7 --numweeks 100 --final_inspection_rate 10 --sampling hybrid --subsamplings xgb/random --weights 0.9/0.1  
```
The example command is to simulate the customs targeting system on a synthetic dataset. The initial training period starts from Jan 1, 2013 `(--train_from)`, and spans 14 days. The last seven days of the training set are held out for validation `(--valid_length)`. With the trained model, customs selection begins on Jan 15 `(--test_from)`. The first testing period spans seven days - batch setting `(--test_length)`. After testing, inspected items are labeled and added to the training set. The simulation terminates after 100 testing periods `(--numweeks)`. The target inspection rate is set as 10% `(--final_inspection_rate)`, which means that 10% of the goods are inspected and levied duties. The hybrid selection strategy consisting of xgb and random is used by 9:1 ratio `(--sampling, --subsamplings, --weights)`. In other words, 9% of the total items are selected by XGBoost for inspection, and the remaining 1% of the items are randomly inspected. 



## Data Format
For your understanding, we upload the synthetic import declarations in the `data/` directory.
Users are expected to preprocess their import declarations in a similar format.
Currently, the framework supports single-item declarations that the target labels; illicitness of the item, revenue by inspection, are marked for each item.
To run the code with real datasets, please refer to `data/` directory. [[README]](./data)

|sgd.id|sgd.date  |importer.id| tariff.code| ... |cif.value|total.taxes|illicit|revenue|
|:----:|:--------:|:---------:|:----------:|:---:|--------:|----------:|:-----:|------:|
| SGD1 | 13-01-02 | IMP826164 |8703241128  | ... |2809     | 647       | 0     | 0     |
| SGD2 | 13-01-02 | IMP837219 |8703232926  | ... |266140   | 3262      | 0     | 0     |
| SGD3 | 13-01-02 | IMP117406 |8517180000  | ... |302275   | 5612      | 0     | 0     | 
| SGD4 | 13-01-02 | IMP435108 |8703222900  | ... |4160     | 514       | 0     | 0     |
| SGD5 | 13-01-02 | IMP717900 |8545200000  | ... |239549   | 397       | 1     | 980   |


## Available Selection Strategies:
### Stand-alone strategies:
```
$ python main.py --sampling random --data synthetic --train_from 20130101 --test_from 20130115 --valid_length 7 --test_length 7 --numweeks 100 --final_inspection_rate 10
$ python main.py --sampling DATE --data real-t --train_from 20150101 --test_from 20150115 --valid_length 7 --test_length 7 --numweeks 300 --initial_inspection_rate 10 --final_inspection_rate 5 --inspection_plan fast_linear_decay --initial_masking importer
$ python main.py --sampling ssl_ae --data real-n --train_from 20130101 --test_from 20130131 --valid_length 7 --test_length 14 --numweeks 100 --initial_inspection_rate 10 --final_inspection_rate 5 --semi_supervised 1
```
#### Supervised strategies (use labeled data only):
* [Random](./query_strategies/random.py): Random selection, often used as a sub-strategy to find novel frauds by compensating the weakness of the selection model based on historical data. 
* [Risky](./query_strategies/risky.py): Simple but effective strategy by using the risky profile indicators (handler's fraud history) to determine the suspiciousness of the trade.
* [XGBoost](./query_strategies/xgb.py): Baseline exploitation strategy using both risky profiles of categorical variables and numeric variables as inputs. [[Reference]](https://xgboost.readthedocs.io/en/latest/python/python_api.html)
* [XBGoost + Logistic Regression](./query_strategies/xgb_lr.py): Two-stage strategy, selection is done by logistic regression results, and logistic regression is done by getting XGB leaf indices.
* [DATE](./query_strategies/DATE.py): Tree-aware dual attentive model for finding the most illicit and valuable imports at the same time; SOTA exploitation strategy. [[Reference]](https://bit.ly/kdd20-date)
* [Tabnet](./query_strategies/tabnet.py): Tabnet trains an encoder-decoder structure to reconstruct masked tabular inputs. Selection is done with binary classification using a trained encoder. [[Reference]](https://github.com/dreamquark-ai/tabnet)
* [Diversity](./query_strategies/diversity.py): Variation of BADGE, this model uses the penultimate layer of the DATE model, then finds centroids by KMeans. In this way, diverse imports can be selected for the next inspection.
* [BADGE](./query_strategies/badge.py): BADGE model uses the gradient embedding of the base model (DATE) and find the most diverse imports by KMeans++. [[Reference]](https://github.com/JordanAsh/badge)
* [bATE](./query_strategies/bATE.py): Proposed model for better exploration. By following the BADGE model, we first use the embeddings of the base model, DATE. Our contribution is to amplify the embedding with extra uncertainty score, and predicted revenue. Finally, we find the most diverse imports by KMeans++. [[Reference]](https://ieeexplore.ieee.org/document/9695316)
* [gATE](./query_strategies/gATE.py): Proposed exploration model, bATE added with gatekeeper. [[Reference]](https://ieeexplore.ieee.org/document/9695316)

#### Semi-supervised strategies (use unlabeled data together, `--semi_supervised 1`):
* [deepSAD](./query_strategies/deepSAD.py): Deep-SAD model, which does semi-supervised anomaly detection by pulling normal-labeled and unlabeled data into a single point, and pushing anomalies away. [[Reference]](https://github.com/lukasruff/Deep-SAD-PyTorch)
* [multideepSAD](./query_strategies/multideepSAD.py): deepSAD variant with several cluster points.
* [SSL-AE](./query_strategies/ssl_ae.py): Semi-supervised learning approach by optimizing reconstruction loss of all imports and binary cross-entropy of labeled imports.


### Hybrid strategies:
```
$ python main.py --sampling hybrid --subsamplings xgb/risky/random --weights 0.7/0.2/0.1 --data synthetic --train_from 20130101 --test_from 20130115 --valid_length 7 --test_length 7 --numweeks 100 --final_inspection_rate 10 
$ python main.py --sampling adahybrid --subsamplings DATE/random --weights 0.9/0.1 --data synthetic --train_from 20130101 --test_from 20130115 --valid_length 7 --test_length 7 --numweeks 100 --final_inspection_rate 10 
$ python main.py --prefix rada-bal-s  --drift pot --mixing reinit --data synthetic --ada_algo ucb --ada_discount decay --ada_lr 3 --ada_epsilon 0.1 --ada_decay 0.9 --sampling rada --subsamplings xgb/random --weights 0.9/0.1 --mode scratch --train_from 20130101 --test_from 20130115 --test_length 7 --valid_length 7 --final_inspection_rate 10 --epoch 10 --numweeks 300
```
* [Hybrid](./query_strategies/hybrid.py): Custom selection with multiple strategies. More than two strategies can be used together.
* [APT-Hybrid](./query_strategies/adahybrid.py): "Adaptive Performance Tuning (APT) Strategy" - Finding the best exploration ratio by using the performance signal. Currently supports two strategies, preferably in the order of exploitation/exploration. The description of this strategy is introduced in Sec 4.1. of our ICDMW 2021 paper [[Link]](https://arxiv.org/pdf/2109.14155.pdf).
* [ADAPT-Hybrid](./query_strategies/radahybrid.py): "Adaptive Drift-Aware and Performance Tuning (ADAPT) Strategy" - Finding the best exploration ratio by using performance signal and drift score. Currently supports two strategies, preferably in the order of exploitation/exploration. The description of this strategy is introduced in Sec 4.3. of our ICDMW 2021 paper [[Link]](https://arxiv.org/pdf/2109.14155.pdf).
* [Drift detectors](./query_strategies/drift.py): Controlling the weights between hybrid subsamplers by measuring the amount of concept drift between the validation and testing set. Currently supports two strategies. In the ICDMW 2021 paper, we mentioned it as "Adaptive Drift-Aware (ADA) strategy" 
  * [POT](./query_strategies/pot.py): Earth-mover distance between the validation- and test- embeddings is used to measure the concept drift. We used a POT library [[Reference]](https://pythonot.github.io/all.html?highlight=emd2#ot.emd2)
  * [P-value](./query_strategies/risky.py): Anderson-Darling test is used to measure the concept drift. [[Reference]](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson_ksamp.html)


## Research
Please find the attached literatures to study. Some of them are uploaded in `./literatures` directory.
* Machine Learning for Customs Fraud Detection [[Link]](https://github.com/YSCHOI-github/Customs_Fraud_Detection): This repository helps practitioners to get used to machine learning for customs fraud detection. This material is worth to check before catching up the DATE paper.
* DATE: Dual-Attentive Tree-aware Embedding for Customs Fraud Detection (KDD'2020) [[Github]](https://bit.ly/kdd20-date): DATE is the Tree-aware dual attentive model for finding the most illicit and valuable imports at the same time. We use DATE as a representative exploitation strategies and its embeddings are used for various strategies. You can test [DATE](./query_strategies/DATE.py) by running `--sampling DATE`.
* Active Learning for Human-in-the-Loop Customs Inspection [[Link]](https://ieeexplore.ieee.org/document/9695316): The key point of this study is that in the conflicting situation between short term revenue and long-term model performance, adding a certain amount of exploration strategy will ensure that the customs targeting system operates sustainably. To that end, our research team proposed an exploration scheme called bATE and gATE, and showed that the model's performance is maintained for a long time when these strategies are used together with existing exploitation strategies. We can easily reproduce this hybrid approach by running [Hybrid](./query_strategies/hybrid.py) with exploitation-exploration pair, such as 90% [DATE](./query_strategies/DATE.py) and 10% [gATE](./query_strategies/gATE.py). 


## Citation
If you find this code useful, please cite the original paper:
```LaTeX
@inproceedings{kimtsai2020date,
  title={DATE: Dual Attentive Tree-aware Embedding for Customs Fraud Detection},
  author={Kim, Sundong and Tsai, Yu-Che and Singh, Karandeep and Choi, Yeonsoo and Ibok, Etim and Li, Cheng-Te and Cha, Meeyoung},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year={2020}
}

@article{kim2021customs,
  title={Active Learning for Human-in-the-Loop Customs Inspection},
  author={Sundong Kim and Tung-Duong Mai and Sungwon Han and Sungwon Park and Thi Nguyen Duc Khanh and Jaechan So and Karandeep Singh and Meeyoung Cha},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year = {2022}
}

@inproceedings{mai2021drift,
  title={{Customs fraud detection in the presence of concept drift}},
  author={Tung-Duong Mai and Kien Hoang and Aitolkyn Baigutanova and Gaukhartas Alina and Sundong Kim},
  booktitle={Proc. of the International Conference on Data Mining Workshops},
  year={2021},
  pages = {370--379},
}
```


## Contribution
We welcome you to contribute to designing new selection strategies, automating feature engineering adaptive to different feature sets, donating anonymized import declarations dataset, and packaging software (PyPI). To collaborate with us, please contact Sundong Kim (sundong@ibs.re.kr). 

