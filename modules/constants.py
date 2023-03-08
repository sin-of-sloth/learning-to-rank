"""
Module that has the various constants used
"""


JAVA_PATH = '/home/arjlal/.jdks/corretto-18.0.2/bin/java'
PERL_PATH = 'usr/bin/perl'

RANKLIB = 'RankLib-2.18.jar'
RANKERS = {
    'MART': 0,
    'LambdaMART': 2,
    'XGBoost': 'XGBoost',
}

MODES = [
    'content_features_only',
    'all_features',
    'weak_labels',
]

DATASETS = ['train', 'vali', 'test']
FOLDS = [1, 2, 3, 4, 5]
TRAIN_METRIC = 'NDCG@10'
TEST_METRICS = ['NDCG@3', 'NDCG@5', 'NDCG@10', 'MAP']

MODEL_SAVE_PATH = 'models'
SCORES_PATH = 'scores'

DATA_FILE = '{dataset_root}/Fold{fold}/{weak_label_prefix}{dataset}.txt'

DATASET_ROOT = 'MSLR-WEB10K'
CONTENT_FEATURES_FILE = f'{DATASET_ROOT}/content_features.txt'
CONTENT_FEATURE_END = 133
CONTENT_FEATURES = [str(feat) for feat in list(range(1, CONTENT_FEATURE_END + 1))]

TRAIN_COMMAND = '{java_path} -jar {ranklib} -train {train_data} -validate {validation_data} -ranker {' \
          'ranker} -metric2t {train_metric} -save {model_filename}'
FEATURE_SET_FLAG = '-feature'
TEST_COMMAND = '{java_path} -jar {ranklib} -load {model} -test {data} -metric2T {test_metric} -idv {score_path}'
