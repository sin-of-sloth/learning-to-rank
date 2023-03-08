from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_svmlight_file, dump_svmlight_file

from modules import constants as const


class WeakLabeler:
    def __init__(self, fold):
        self.fold = fold
        self.datasets = {
            dataset: self.load_mslr_data(fold, dataset)
            for dataset in const.DATASETS
        }

    def create_weak_labels(self):
        train_x = self.datasets['train']['x'][:, const.CONTENT_FEATURE_END:]
        train_y = self.datasets['train']['y']
        rf = RandomForestClassifier(max_depth=5, n_estimators=25)
        rf.fit(train_x, train_y)

        for dataset in self.datasets:
            weak_y = rf.predict(self.datasets[dataset]['x'][:, const.CONTENT_FEATURE_END:])
            weak_file = self.get_weak_file_name(self.fold, dataset)
            dump_svmlight_file(
                X=self.datasets[dataset]['x'],
                y=weak_y,
                f=weak_file,
                zero_based=False,
                query_id=self.datasets[dataset]['qid']
            )

    @staticmethod
    def load_mslr_data(fold, dataset):
        data_file = const.DATA_FILE.format(
            dataset_root=const.DATASET_ROOT,
            fold=fold,
            weak_label_prefix='',
            dataset=dataset
        )
        x, y, qid = load_svmlight_file(data_file, query_id=True)
        return {'x': x, 'y': y, 'qid': qid}

    @staticmethod
    def get_weak_file_name(fold, dataset):
        """
        returns the path of the file to save weak-labeled data
        """
        return const.DATA_FILE.format(
            dataset_root=const.DATASET_ROOT,
            fold=fold,
            weak_label_prefix='weak_',
            dataset=dataset
        )
