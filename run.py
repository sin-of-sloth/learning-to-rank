from modules.ranker import MSLRRanker
from modules.weak_labeler import WeakLabeler
from modules import constants as const
from modules import utils


def main():
    # create weak label data
    for fold in const.FOLDS:
        weak_labeler = WeakLabeler(fold)
        weak_labeler.create_weak_labels()
    # do ranking
    ranker = MSLRRanker()
    for mode in const.MODES:
        ranker.rank_mslr(mode)


def init():
    """
    Initializes empty directories to save scores and models
    """
    for mode in const.MODES:
        for ranker in const.RANKERS:
            model_dir = f'{const.MODEL_SAVE_PATH}/{mode}/{ranker}'
            utils.make_dir(model_dir)

            for metric in const.TEST_METRICS:
                score_dir = f'{const.SCORES_PATH}/{mode}/{ranker}/{metric}'
                utils.make_dir(score_dir)

    # create feature set file for only content features (used in ranklib)
    utils.write_list_to_file(const.CONTENT_FEATURES_FILE, const.CONTENT_FEATURES)


def print_final_scores():
    for mode in const.MODES:
        print(f'\n====================== {mode} ======================\n')
        for ranker_name in const.RANKERS.keys():
            print(f'{ranker_name}')
            for dataset in const.DATASETS:
                for metric in const.TEST_METRICS:
                    try:
                        fold_scores = []
                        for fold in const.FOLDS:
                            score_file = f'{const.SCORES_PATH}/{mode}/{ranker_name}/{metric}/{ranker_name}_score_{fold}_{dataset}_{metric}.txt'
                            scores = utils.read_file_to_list(score_file)
                            all_score = list(filter(lambda s: 'all' in s, scores))[0]
                            fold_scores.append(float(all_score.split(' ')[-1]))
                        avg_score = sum(fold_scores) / len(fold_scores)
                        print(f'\t\t{dataset}\t\t{metric}\t\t{avg_score:.8f}')
                    except FileNotFoundError:
                        pass


if __name__ == '__main__':
    init()
    main()
    print_final_scores()
