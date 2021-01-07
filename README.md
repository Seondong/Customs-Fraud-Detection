# Simulation Framework for Customs Selection 

Use your collected declarations data, fit the provided models, find the best selection strategy for your customs. 
This framework supports general import declarations. 



## How to Install  

1. Setup your Python environment: e.g., Anaconda Python 3.7 [[Guide]](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
```
$ source activate py37 
```

2. Install requirements 
```
$ pip install -r requirements.txt
# Please install the Ranger optimizer by following its instruction.
```

3. Run the codes: Refer to main.py for hyperparameters, .sh files in `./bash` directory will give you some ideas how to run codes effectively. 
```
$ export CUDA_VISIBLE_DEVICES=3 && python main.py --data real-m --semi_supervised 0 --batch_size 128 --sampling hybrid --subsamplings bATE/DATE --weights 0.1/0.9 --mode scratch --train_from 20130101 --test_from 20130701 --test_length 30 --valid_length 30 --initial_inspection_rate 20 --final_inspection_rate 5 --epoch 5 --closs bce --rloss full --save 0 --numweeks 100 --inspection_plan direct_decay
```

4. (For IBS server users) If you want to use GPU on IBS server, you have to load a cuda module before running your code. 
```
$ ssh node01 (or ssh node02)
$ module load cuda/10.0
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

For team: to run this code with real datasets (real-n, real-m, real-t, real-c), please sign the [Confidentiality Statement](./data/Confidentiality_Statement.docx) and submit to Sundong Kim (sundong@ibs.re.kr). He will share you the location of the datasets. Members should only access the datasets through the IBS server, and MUST NOT keep them.

|Data       |# Imports |Illicit rate| Period           | Misc          |
|:---------:|:--------:|:----------:|:----------------:|:-------------:|
| synthetic | 0.1M     | 7.58%      | Jan 13 – Dec 13  |[With the repo](./data/synthetic-imports-declarations.csv)|
| real-m    | 0.42M    | 1.64%      | Jan 13 – Dec 16  |Confidential   |
| real-c    | 1.90M    | 1.21%      | Jan 16 – Dec 19  |Confidential   |
| real-n    | 1.93M    | 4.12%      | Jan 13 – Dec 17  |Confidential   |
| real-t    | 4.17M    | 8.16%      | Jan 15 – Dec 19  |Confidential   |



## Available Selection Strategies:
* [Random](./query_strategies/random.py): Random selection, often used as a sub-strategy to find novel frauds by compensating the weakness of the selection model based on historical data. 
* [XGBoost](./query_strategies/xgb.py): Baseline exploitation strategy using both risky profiles of categorical variables and numeric variables as inputs. [[Reference]](https://xgboost.readthedocs.io/en/latest/python/python_api.html)
* [XBGoost + Logistic Regression](./query_strategies/xgb_lr.py): Two-stage strategy, selection is done by logistic regression results, and logistic regression is done by getting XGB leaf indices.
* [DATE](./query_strategies/DATE.py): Tree-aware dual attentive model for finding the most illicit and valuable imports at the same time; SOTA exploitation strategy. [[Reference]](https://bit.ly/kdd20-date)
* [Tabnet](./query_strategies/tabnet.py): Tabnet trains an encoder-decoder structure to reconstruct masked tabular inputs. Selection is done with binary classification using a trained encoder. [[Reference]](https://github.com/dreamquark-ai/tabnet)
* [Diversity](./query_strategies/diversity.py): Variation of BADGE, this model uses the penultimate layer of the DATE model, then finds centroids by KMeans. In this way, diverse imports can be selected for the next inspection.
* [BADGE](./query_strategies/badge.py): BADGE model uses the gradient embedding of the base model (DATE) and find the most diverse imports by KMeans++. [[Reference]](https://github.com/JordanAsh/badge)
* [bATE](./query_strategies/bATE.py): Proposed model for better exploration. By following the BADGE model, we first use the embeddings of the base model, DATE. Our contribution is to amplify the embedding with extra uncertainty score, and predicted revenue. Finally, we find the most diverse imports by KMeans++. [[Reference]](https://arxiv.org/abs/2010.14282)
* [gATE](./query_strategies/gATE.py): Proposed exploration model, bATE added with gatekeeper. [[Reference]](https://arxiv.org/abs/2010.14282)
* [deepSAD](./query_strategies/deepSAD.py): Deep-SAD model, which does semi-supervised anomaly detection by pulling normal-labeled and unlabeled data into a single point, and pushing anomalies away. [[Reference]](https://github.com/lukasruff/Deep-SAD-PyTorch)
* [multideepSAD](./query_strategies/multideepSAD.py): deepSAD variant with several cluster points.
* [SSL-AE](./query_strategies/ssl_ae.py): Semi-supervised learning approach by optimizing reconstruction loss of all imports and binary cross-entropy of labeled imports.
* [Hybrid](./query_strategies/hybrid.py): Support mixing several strategies. (Adahybrid method can adaptively change the exploration rate)
* [Adahybrid](./query_strategies/hybrid.py): Adaptively changing exploration ratio of the hybrid strategy. Aim to tackle exploitation-exploration dilemma in smarter way. The description of this strategy is introduced in page 21-22 of John's report [[PDF]](./literatures/URP_Report_TungDuongMai.pdf).



## Brief Introduction of Our Research Directions
Please find the attached literatures to study. Some of them are uploaded in `./literatures` directory.
* Machine Learning for Customs Fraud Detection [[Link]](https://github.com/YSCHOI-github/Customs_Fraud_Detection): This repository helps practitioners to get used to machine learning for customs fraud detection. This material is worth to check before catching up the DATE paper.
* DATE: Dual-Attentive Tree-aware Embedding for Customs Fraud Detection (KDD'2020) [[Github]](https://bit.ly/kdd20-date) [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3394486.3403339) [[Slides]](http://seondong.github.io/assets/papers/2020_KDD_DATE_slides.pdf) [[Presentation (20 min)]](https://youtu.be/S-29rTbvH6c) [[Promotional video]](https://youtu.be/YhfxCHBNM2g): DATE is the Tree-aware dual attentive model for finding the most illicit and valuable imports at the same time. In the 'Take a Chance' paper, we use DATE as a representative of exploitation strategies. We are building algorithms on top of DATE architecture. We can easily reproduce DATE by running [DATE](./query_strategies/DATE.py) on this repository.
* Take a Chance: Managing the Exploitation-Exploration Dilemma in Customs Fraud Detection via Online Active Learning [[Link]](https://arxiv.org/abs/2010.14282): The key point of this study is that in the conflicting situation between short term revenue and long-term model performance, adding a certain amount of exploration strategy will ensure that the customs targeting system operates sustainably. To that end, our research team proposed an exploration scheme called bATE and gATE, and showed that the model's performance is maintained for a long time when these strategies are used together with existing exploitation strategies. We can easily reproduce this hybrid approach by running [Hybrid](./query_strategies/hybrid.py) with exploitation-exploration pair, such as 90% [DATE](./query_strategies/DATE.py) and 10% [gATE](./query_strategies/gATE.py). The comments that we received from the reviewers of The Web Conference 2021 can be found here. [PDF](./literatures/Reviews_and_rebuttals_TheWebConf2021.pdf) 
* This proposal [[PDF]](./literatures/YSF_proposal_Sundong_Lifelong_tabular_learning.pdf) introduces several research directions that our team is pursuing in a long term.


## Contribution
We welcome you to contribute to designing new selection strategies, automating feature engineering adaptive to different feature sets, donating anonymized import declarations dataset, and packaging software (PyPI). To closely work with us, please contact Sundong Kim (sundong@ibs.re.kr). We often hire interns whenever we have interesting ideas to develop together. KAIST students can also work with us by applying independent studies and URP programs, but please contact Sundong and send your application through below link. We expect you to have strong analytical background and coding skills. [[Application Link]](https://docs.google.com/forms/d/e/1FAIpQLSeoLB0DI_MET1pRuQu5dh-HIUaVwvr3CcGziL03_cPDC5HfCw/viewform)

