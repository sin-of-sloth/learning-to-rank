"""
Module that calls the necessary rankers on the MSLR-WEB10K dataset
"""
import os
from sklearn.datasets import load_svmlight_file
from xgboost import XGBRanker
import numpy as np

from modules import constants as const
from modules import utils
from modules.scorers import ndcg_at_k, mean_average_precision


class MSLRRanker:
    """
    Class that calls the rankers for each fold in the dataset
    """
    @staticmethod
    def rank_mslr(mode):
        for ranker_name, ranker in const.RANKERS.items():
            for fold in const.FOLDS:
                if ranker_name == 'XGBoost':
                    xgb_ranker = XGBoostRanker()
                    xgb_ranker.rank(fold, mode, ranker_name)
                else:
                    # use the ranklib jar for ranking
                    ranklib_ranker = RankLibRanker()
                    ranklib_ranker.rank(ranker, ranker_name, mode, fold)


class XGBoostRanker:
    """
    Class that ranks data using XGBoost
    """
    def rank(self, fold, mode, ranker_name):
        data = self.get_xgboost_fold_data(fold, mode)
        self.train_xbg_ranker(data, fold, mode, ranker_name)
        self.test_xbg_ranker(data, fold, mode, ranker_name)

    @staticmethod
    def train_xbg_ranker(data, fold, mode, ranker_name):
        xgb_model = XGBRanker(n_estimators=5, max_depth=5, objective='rank:ndcg', verbosity=2)
        params = {
            'eval_metric': 'ndcg'
        }
        xgb_model.set_params(**params)
        xgb_model.fit(
            X=data['train']['x'],
            y=data['train']['y'],
            group=data['train']['groups'],
            eval_set=[(data['vali']['x'], data['vali']['y'])],
            eval_group=[data['vali']['groups']],
            verbose=True,
            # qid=data['train']['qid'],
        )

        model_file = f'{const.MODEL_SAVE_PATH}/{mode}/{ranker_name}/{ranker_name}_model_{fold}.txt'
        xgb_model.save_model(model_file)

    @staticmethod
    def test_xbg_ranker(data, fold, mode, ranker_name):
        model_file = f'{const.MODEL_SAVE_PATH}/{mode}/{ranker_name}/{ranker_name}_model_{fold}.txt'
        xgb_model = XGBRanker()
        xgb_model.load_model(model_file)

        for dataset in data:
            # predict for each group of qids
            i = 0
            metric_data = {
                metric: {
                    'file': f'{const.SCORES_PATH}/{mode}/{ranker_name}/{metric}/{ranker_name}_score_{fold}_{dataset}_{metric}.txt',
                    'score_strs': [],
                    'scores': [],
                }
                for metric in const.TEST_METRICS
            }
            for group_size in data[dataset]['groups']:
                pred_y = xgb_model.predict(data[dataset]['x'][i:(i + group_size)])
                for metric in const.TEST_METRICS:
                    if 'NDCG' in metric:
                        ndcg_k = int(metric.split('@')[1])
                        score = ndcg_at_k(pred_y, ndcg_k)
                    else:
                        # not using MAP for XGBoost ranking
                        pass
                    metric_data[metric]['score_strs'].append(f'{metric}   {data[dataset]["qid"][i]}   {score}')
                    metric_data[metric]['scores'].append(score)
                i += group_size

            for metric in const.TEST_METRICS:
                avg_score = sum(metric_data[metric]['scores']) / len(metric_data[metric]['scores'])
                metric_data[metric]['score_strs'].append(f'{metric}   all   {avg_score}')
                print(f'fold{fold} {dataset} {metric} {avg_score}')
                utils.write_list_to_file(metric_data[metric]['file'], metric_data[metric]['score_strs'])

    def get_xgboost_fold_data(self, fold, mode):
        return {
            dataset: self.load_xgboost_data_file(fold, dataset, mode)
            for dataset in const.DATASETS
        }

    def load_xgboost_data_file(self, fold, dataset, mode):
        data_file = const.DATA_FILE.format(
            dataset_root=const.DATASET_ROOT,
            fold=fold,
            weak_label_prefix='weak_' if mode == 'weak_labels' else '',
            dataset=dataset
        )
        x, y, qids = load_svmlight_file(data_file, query_id=True)
        return {
            'x': x[:, :const.CONTENT_FEATURE_END] if mode == 'content_features_only' else x,
            'y': y,
            'qid': qids,
            'groups': self.get_group_data(qids)
        }

    @staticmethod
    def get_binary_relevance(rels):
        return [0 if rel < 1 else 1 for rel in rels]

    @staticmethod
    def get_group_data(qids):
        group_data = []
        i = -1
        curr_group = None
        for qid in qids:
            if qid != curr_group:
                i += 1
                group_data.append(1)
                curr_group = qid
            else:
                group_data[i] += 1

        return np.asarray(group_data)


class RankLibRanker:
    """
    Class that ranks data using RankLib
    """
    def rank(self, ranker, ranker_name, mode, fold):
        data = self.get_ranklib_fold_data(fold, mode == 'weak_labels')
        model = self.train_ranklib_model(
            ranker,
            ranker_name,
            fold,
            data['train'],
            data['vali'],
            mode
        )
        # test the model on actual data
        test_data = self.get_ranklib_fold_data(fold, weak_labels=False)
        self.test_ranklib_model(model, ranker_name, fold, test_data, mode)

    @staticmethod
    def train_ranklib_model(ranking_method, ranker_name, fold, train_data, validate_data, mode):
        model = f'{const.MODEL_SAVE_PATH}/{mode}/{ranker_name}/{ranker_name}_model_{fold}.txt'
        # create the command to be run for training the model using ranklib
        train_cmd = const.TRAIN_COMMAND.format(
            java_path=const.JAVA_PATH,
            ranklib=const.RANKLIB,
            train_data=train_data,
            validation_data=validate_data,
            ranker=ranking_method,
            train_metric=const.TRAIN_METRIC,
            model_filename=model
        )
        # if we want to train using only the content features,
        # add the feature flag to the command
        if mode == 'content_features_only':
            train_cmd = f'{train_cmd} {const.FEATURE_SET_FLAG} {const.CONTENT_FEATURES_FILE}'

        os.system(train_cmd)

        return model

    @staticmethod
    def test_ranklib_model(model, ranker_name, fold, datasets, mode):
        for dataset, data in datasets.items():
            for metric in const.TEST_METRICS:
                score = f'{const.SCORES_PATH}/{mode}/{ranker_name}/{metric}/{ranker_name}_score_{fold}_{dataset}_{metric}.txt'
                os.system(
                    const.TEST_COMMAND.format(
                        java_path=const.JAVA_PATH,
                        ranklib=const.RANKLIB,
                        model=model,
                        data=data,
                        test_metric=metric,
                        score_path=score
                    )
                )

    @staticmethod
    def get_ranklib_fold_data(fold, weak_labels):
        return {
            dataset: const.DATA_FILE.format(
                dataset_root=const.DATASET_ROOT,
                weak_label_prefix='weak_' if weak_labels else '',
                fold=fold,
                dataset=dataset,
            )
            for dataset in const.DATASETS
        }
