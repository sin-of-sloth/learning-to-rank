# Learning-to-Rank

<p>
    <a href="https://www.python.org/downloads/release/python-3106/">
        <img src="https://img.shields.io/static/v1?label=python&style=flat-square&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48IS0tIFVwbG9hZGVkIHRvOiBTVkcgUmVwbywgd3d3LnN2Z3JlcG8uY29tLCBHZW5lcmF0b3I6IFNWRyBSZXBvIE1peGVyIFRvb2xzIC0tPgo8c3ZnIHdpZHRoPSI4MDBweCIgaGVpZ2h0PSI4MDBweCIgdmlld0JveD0iMCAwIDMyIDMyIiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPg0KPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMy4wMTY0IDJDMTAuODE5MyAyIDkuMDM4MjUgMy43MjQ1MyA5LjAzODI1IDUuODUxODVWOC41MTg1MkgxNS45MjM1VjkuMjU5MjZINS45NzgxNEMzLjc4MTA3IDkuMjU5MjYgMiAxMC45ODM4IDIgMTMuMTExMUwyIDE4Ljg4ODlDMiAyMS4wMTYyIDMuNzgxMDcgMjIuNzQwNyA1Ljk3ODE0IDIyLjc0MDdIOC4yNzMyMlYxOS40ODE1QzguMjczMjIgMTcuMzU0MiAxMC4wNTQzIDE1LjYyOTYgMTIuMjUxNCAxNS42Mjk2SDE5LjU5NTZDMjEuNDU0NyAxNS42Mjk2IDIyLjk2MTcgMTQuMTcwNCAyMi45NjE3IDEyLjM3MDRWNS44NTE4NUMyMi45NjE3IDMuNzI0NTMgMjEuMTgwNyAyIDE4Ljk4MzYgMkgxMy4wMTY0Wk0xMi4wOTg0IDYuNzQwNzRDMTIuODU4OSA2Ljc0MDc0IDEzLjQ3NTQgNi4xNDM3OCAxMy40NzU0IDUuNDA3NDFDMTMuNDc1NCA0LjY3MTAzIDEyLjg1ODkgNC4wNzQwNyAxMi4wOTg0IDQuMDc0MDdDMTEuMzM3OCA0LjA3NDA3IDEwLjcyMTMgNC42NzEwMyAxMC43MjEzIDUuNDA3NDFDMTAuNzIxMyA2LjE0Mzc4IDExLjMzNzggNi43NDA3NCAxMi4wOTg0IDYuNzQwNzRaIiBmaWxsPSJ1cmwoI3BhaW50MF9saW5lYXJfODdfODIwNCkiLz4NCjxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNMTguOTgzNCAzMEMyMS4xODA1IDMwIDIyLjk2MTYgMjguMjc1NSAyMi45NjE2IDI2LjE0ODJWMjMuNDgxNUwxNi4wNzYzIDIzLjQ4MTVMMTYuMDc2MyAyMi43NDA4TDI2LjAyMTcgMjIuNzQwOEMyOC4yMTg4IDIyLjc0MDggMjkuOTk5OCAyMS4wMTYyIDI5Ljk5OTggMTguODg4OVYxMy4xMTExQzI5Ljk5OTggMTAuOTgzOCAyOC4yMTg4IDkuMjU5MjggMjYuMDIxNyA5LjI1OTI4TDIzLjcyNjYgOS4yNTkyOFYxMi41MTg1QzIzLjcyNjYgMTQuNjQ1OSAyMS45NDU1IDE2LjM3MDQgMTkuNzQ4NSAxNi4zNzA0TDEyLjQwNDIgMTYuMzcwNEMxMC41NDUxIDE2LjM3MDQgOS4wMzgwOSAxNy44Mjk2IDkuMDM4MDkgMTkuNjI5Nkw5LjAzODA5IDI2LjE0ODJDOS4wMzgwOSAyOC4yNzU1IDEwLjgxOTIgMzAgMTMuMDE2MiAzMEgxOC45ODM0Wk0xOS45MDE1IDI1LjI1OTNDMTkuMTQwOSAyNS4yNTkzIDE4LjUyNDQgMjUuODU2MiAxOC41MjQ0IDI2LjU5MjZDMTguNTI0NCAyNy4zMjkgMTkuMTQwOSAyNy45MjU5IDE5LjkwMTUgMjcuOTI1OUMyMC42NjIgMjcuOTI1OSAyMS4yNzg1IDI3LjMyOSAyMS4yNzg1IDI2LjU5MjZDMjEuMjc4NSAyNS44NTYyIDIwLjY2MiAyNS4yNTkzIDE5LjkwMTUgMjUuMjU5M1oiIGZpbGw9InVybCgjcGFpbnQxX2xpbmVhcl84N184MjA0KSIvPg0KPGRlZnM+DQo8bGluZWFyR3JhZGllbnQgaWQ9InBhaW50MF9saW5lYXJfODdfODIwNCIgeDE9IjEyLjQ4MDkiIHkxPSIyIiB4Mj0iMTIuNDgwOSIgeTI9IjIyLjc0MDciIGdyYWRpZW50VW5pdHM9InVzZXJTcGFjZU9uVXNlIj4NCjxzdG9wIHN0b3AtY29sb3I9IiMzMjdFQkQiLz4NCjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzE1NjVBNyIvPg0KPC9saW5lYXJHcmFkaWVudD4NCjxsaW5lYXJHcmFkaWVudCBpZD0icGFpbnQxX2xpbmVhcl84N184MjA0IiB4MT0iMTkuNTE5IiB5MT0iOS4yNTkyOCIgeDI9IjE5LjUxOSIgeTI9IjMwIiBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+DQo8c3RvcCBzdG9wLWNvbG9yPSIjRkZEQTRCIi8+DQo8c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiNGOUM2MDAiLz4NCjwvbGluZWFyR3JhZGllbnQ+DQo8L2RlZnM+DQo8L3N2Zz4=&message=3.10.6&color=green" />
    </a>
    <a href="https://sourceforge.net/p/lemur/wiki/RankLib/">
        <img src="https://img.shields.io/static/v1?label=RankLib&style=flat-square&message=2.18&color=green" />
    </a>
    <a href="https://github.com/dmlc/xgboost">
        <img src="https://img.shields.io/static/v1?label=XGBoost&style=flat-square&message=1.7.4&color=green" />
    </a>
</p>

Done as part of CS-572: Information Retrieval course instructed by
[Dr. Eugene Agichtein](http://www.cs.emory.edu/~eugene/).

Developed and tested on Ubuntu 22.04.2 LTS.

## Contents
1. [Background Information](#1-background-information)
2. [Implementation Details](#2-implementation-details)
3. [Try it yourself](#3-try-it-yourself)

## 1 Background Information

### 1.1 Dataset

[MSLR-WEB10K](https://www.microsoft.com/en-us/research/project/mslr/) - the dataset is machine learning data, in which
queries and urls are represented by IDs. The datasets consist of feature vectors extracted from query-url pairs along
with relevance judgment labels.

### 1.2 Implemented Ranking Methods

- `MART`
- `LambdaMART`
- `XGBoost`

`MART`, `LambdaMart` implementation from `RankLib`: [https://github.com/codelibs/ranklib](https://github.com/codelibs/ranklib)
or [https://sourceforge.net/p/lemur/wiki/RankLib/](https://sourceforge.net/p/lemur/wiki/RankLib/)

`XGBoost`:  [https://github.com/dmlc/xgboost](https://github.com/dmlc/xgboost) 

### 1.3 Dataset Variations

The models are trained and predicted on the provided 5 folds on 3 variations of the MSLR-WEB10K dataset.

- **Setting 1:**

    Supervised ranking with Content Features Only (features [1-133] in the dataset).
- **Setting 2:**

    Supervised ranking with full feature set (include behavior features) (134-136).
- **Setting 3:**

    Partially supervised ranking using click features as weak labels. Derive noisy/weak relevance labels using click
features (click count, dwell time), and use those derived labels to train the three models.

    _Example labeling heuristic:_
query-url click fraction for URL is > 0.5 of all clicks for query, and average dwell time > 10 seconds => relevant. 

    Experiment with other ideas for inferring relevance from the (limited) click data available. For example, consider
*learning* to predict the relevance label from click data features.

### 1.4 Evaluation Metrics

- `NDCG@3`
- `NDCG@5`
- `NDCG@10`
- `MAP`

## 2 Implementation Details

### 2.1 Models

#### 2.1.1 `MART` and `LambdaMART`

Both `MART` and `LambdaMART` were implemented by invoking
[`RankLib-2.18.jar`](https://sourceforge.net/projects/lemur/files/lemur/RankLib-2.18/) from a python script using
RankLib’s default parameters. The parameters used are:

| Parameter                                         | Value   |
|:--------------------------------------------------|:--------|
| Metric to optimize on training data, `metric2t`   | NDCG@10 |
| Number of trees                                   | 100     |
| Number of leaves for each tree                    | 10      |
| Learning rate                                     | 0.1     |
| Number of threshold candidates for tree splitting | 256     |
| Minimum # samples each leaf has to contain        | 1       |
| Early stopping rounds on validation               | 100     |

The models were evaluated on the required metrics using the jar file as well.

Sample command to train a model and save it:
```
$ java -jar RankLib-2.18.jar -train MSLR-WEB10K/Fold<fold>/train.txt -validate MSLR-WEB10K/Fold<fold>/vali.txt -ranker <ranker> -metric2t NDCG@10 -save <model_save_path>
```

`<ranker>` takes values `0` for `MART` and `2` for `LambdaMART`.

_NOTE: To train a model on a subset of features, create a feature subset file where the list of features to be
considered by the learner are specified, each on a separate line. Provide the argument `-feature <feature_subset_file>`
to the command._

Sample command to evaluate a pre-trained model and save the model performance:
```
$ java -jar RankLib-2.18.jar -load <model_save_path> -test MSLR-WEB10K/Fold<fold>/test.txt -metric2T <test_metric> -idv <score_save_path>
```

For more information on how to use, see
[The Lemur Project / Wiki / How to Use](https://sourceforge.net/p/lemur/wiki/RankLib%20How%20to%20use/).

#### 2.1.2 `XGBoost`

`XGBoost` implemented using its python package. Used the model
[`xgboost.XGBRanker`](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRanker) with the
following parameters:

| Parameter                                                     | Value       |
|:--------------------------------------------------------------|:------------|
| Number of gradient boosted trees, `n_estimators`              | 5           |
 | Maximum tree depth for base learners, `max_depth`             | 5           |
 | Learning objective, `objective`                               | `rank:ndcg` |
 | Metric used for monitoring the training result, `eval_metric` | `ndcg`      |

The models were evaluated using a custom function to get the ndcg values; MAP was not evaluated for the models.

### 2.2 Creating Weak Relevance Labels from Click Data

Weak relevance labels for each fold was created by using 
[`sklearn.ensemble.RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
For each fold of data, a model was created using only the click features from the training data. New datasets were
created for train, validation, and test, where the relevance labels were predicted using the generated model using their
click features in the same format as MSLR-WEB10K data. The parameters used for the model are
`{max_depth=5, n_estimators=25}`.

### 2.3 Files and Directories

- `run.py` - driver program that calls required methods
- `modules/` - has all the required classes
- Fetch the MSLR-WEB10K data from [here](https://1drv.ms/u/s!AtsMfWUz5l8nbOIoJ6Ks0bEMp78) and place it in the root
directory of this repo, so that a dataset can be accessed from the path `MSRL-WEB10L/Fold<fold>/<dataset>.txt`
- Requirements can be found in `requirements.txt`
- `model_scores.pdf` - sample set of scores averaged across folds for each model and setting
- Models generated will be saved to the path `models/<setting>/<ranker>/<ranker>_model_<fold>.txt`
- Scores will be saved to the path `scores/<setting>/<ranker>/<metric>/<ranker>_score_<fold>_<dataset>_<metric>.txt`

`<setting>` can be one of `content_features_only` / `all_features` / `weak_labels`

`<ranker>` can be one of `MART` / `LambdaMART` / `XGBoost`

`<fold>` can be one of `1` / `2` / `3` / `4` / `5`

`<metric>` can be one of `NDCG@3` / `NDCG@5` / `NDCG@10` / `MAP`

`<dataset>` can be one of `train` / `vali` / `test`

### 2.4 Code Flow

- Initializes empty directories to save scores and models
- Creates the feature subset file containing only the content features
- Creates weak-labeled data as mentioned in **Setting 3** in [Dataset Variations](#13-dataset-variations) for each fold
and saves the new datasets to the path `MSLR-WEB10K/Fold<fold>/weak_<dataset>.txt`
- Creates the models and evaluates their performances for the required metrics
- Prints the final scores averaged across the five folds for each model

## 3. Try it yourself

### 3.1 Installing Requirements

From the root of the repo:

- Create a virtual environment if you'd like and activate it:
    ```
    $ virtualenv -p python3 .venv
    $ source .venv/bin/activate
    ```

- Install the requirements:
    ```
    $ pip3 install -r requirements.txt
    ```

- Download [MSLR-WEB10K data](https://1drv.ms/u/s!AtsMfWUz5l8nbOIoJ6Ks0bEMp78) and
[RankLib-2.18 jar](https://sourceforge.net/projects/lemur/files/lemur/RankLib-2.18/) to the root of the repo

### 3.2 Run the program

- Run the program
    ```
    $ python3 run.py
    ```

You can try creating your own weak labels by modifying the code in [`modules/weak_labeler.py`](./modules/weak_labeler.py).
