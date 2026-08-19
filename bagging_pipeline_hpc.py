import tpot
import traceback
import dill as pickle
import os
import numpy as np
from estimator_node_gradual import EstimatorNodeGradual
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

from ConfigSpace import ConfigurationSpace, Integer
from tpot.search_spaces.pipelines import SequentialPipeline, WrapperPipeline
from row_sample import RowSampler

import warnings
warnings.filterwarnings('ignore')

# defines a constrained search space with only three steps
def get_pipeline_space(seed):
    return tpot.search_spaces.pipelines.SequentialPipeline([
        tpot.config.get_search_space(
            ["selectors_classification", "Passthrough"], random_state=seed, base_node=EstimatorNodeGradual),
        tpot.config.get_search_space(
            ["transformers", "Passthrough"], random_state=seed, base_node=EstimatorNodeGradual),
        tpot.config.get_search_space("classifiers", random_state=seed, base_node=EstimatorNodeGradual)])


# custom search space with row sampling
def get_bagging_pipeline_space(seed):
    inner_pipeline = SequentialPipeline([
        tpot.config.get_search_space(
            ["selectors_classification", "Passthrough"], random_state=seed, base_node=EstimatorNodeGradual),
        tpot.config.get_search_space(
            ["transformers", "Passthrough"], random_state=seed, base_node=EstimatorNodeGradual),
        tpot.config.get_search_space(
            "classifiers", random_state=seed, base_node=EstimatorNodeGradual),
    ])

    row_sampler_configspace = ConfigurationSpace(
        space={
            "random_state": Integer("random_state", bounds=(0, 10_000)),
        }
    )

    return WrapperPipeline(
        method=RowSampler,
        space=row_sampler_configspace,
        estimator_search_space=inner_pipeline,
    )



def get_cv_predictions(estimator, X_train, y_train, cv_splits, random_state):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    cv_preds = np.empty(len(y_train), dtype=int)

    for train_idx, valid_idx in cv.split(X_train, y_train):
        est_clone = clone(estimator)

        try:
            est_clone.fit(X_train[train_idx], y_train[train_idx])
            cv_preds[valid_idx] = est_clone.predict(X_train[valid_idx])
        except Exception:
            print('pipeline failed')

    return cv_preds


def get_cv_probas(estimator, X_train, y_train, cv_splits, random_state):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    n_classes = len(np.unique(y_train))
    cv_probas = np.zeros((len(y_train), n_classes))

    for train_idx, valid_idx in cv.split(X_train, y_train):
        est_clone = clone(estimator)

        try:
            est_clone.fit(X_train[train_idx], y_train[train_idx])
            cv_probas[valid_idx] = est_clone.predict_proba(X_train[valid_idx])
        except Exception:
            print('pipeline failed')
    return cv_probas



def clean_eval_inds(eval_inds):
    # filter out the broken pipelines
    filtered_eval_inds = eval_inds[eval_inds["roc_auc_score"].notna()]
    print("length of filtered eval_inds:", len(filtered_eval_inds))
    return filtered_eval_inds


def greedy_forward_search(filtered_eval_inds, X_train, y_train, seed):
    n_classes = len(np.unique(y_train))

    estimators = filtered_eval_inds.iloc[:, 10].tolist()

    est_cv_probas = {est: get_cv_probas(est, X_train, y_train, cv_splits=5, random_state=seed) for est in estimators}

    # remove bad estimators (pipeline failed during CV)
    failed = [est for est, probas in est_cv_probas.items() if np.all(probas == 0)]
    if failed:
        print(f"dropping {len(failed)} estimators with failed CV probas")
    est_cv_probas = {est: probas for est, probas in est_cv_probas.items() if not np.all(probas == 0)}
    estimators = list(est_cv_probas.keys())

    temp_ensemble = []

    for _ in range(50):
        best_candidate = None
        best_candidate_auroc = 0


        for est in estimators:

            # test ensemble CV accuracy when each candidate is added
            candidate_probas = [est_cv_probas[e] for e in temp_ensemble + [est]]
            temp_probas = combine_probas(candidate_probas)

            if n_classes == 2:
                temp_auroc = roc_auc_score(y_train, temp_probas[:, 1])
            else:
                temp_auroc = roc_auc_score(y_train, temp_probas, multi_class="ovr", average="macro")


            if temp_auroc > best_candidate_auroc:
                best_candidate = est
                best_candidate_auroc = temp_auroc

        print(f"ensemble acc changes to {best_candidate_auroc:.4f}")
        temp_ensemble.append(best_candidate)


    print(f"FINAL ensemble size: {len(temp_ensemble)}")

    final_ensemble = [est.fit(X_train, y_train) for est in temp_ensemble]
    return final_ensemble



def vote_soft(estimators, X_test, weights=None):
    probas = np.stack([est.predict_proba(X_test) for est in estimators])
    if weights is not None:
        weights = np.asarray(weights).reshape(-1, 1, 1)
        probas *= weights

    return np.argmax(probas.sum(axis=0), axis=1)

def vote_soft_proba(estimators, X_test, weights=None):
    probas = np.stack([est.predict_proba(X_test) for est in estimators])

    if weights is not None:
        weights = np.asarray(weights).reshape(-1, 1, 1)
        return np.sum(probas * weights, axis=0) / np.sum(weights)

    return np.mean(probas, axis=0)

def combine_preds(proba_list, weights=None):
    probas = np.stack(proba_list, axis=0)

    if weights is not None:
        weights = np.asarray(weights).reshape(-1, 1, 1)  # (n_models, 1, 1)
        probas = probas * weights

    # Average (or weighted sum) across models
    avg_proba = probas.sum(axis=0) / (weights.sum() if weights is not None else len(proba_list))

    # Pick the class with max probability
    return np.argmax(avg_proba, axis=1)

def combine_probas(proba_list, weights=None):
    probas = np.stack(proba_list, axis=0)
    if weights is not None:
        weights = np.asarray(weights).reshape(-1, 1, 1)
        probas = probas * weights
    return probas.sum(axis=0) / (weights.sum() if weights is not None else len(proba_list))


def main():
    parser = argparse.ArgumentParser()
    # number of threads
    parser.add_argument("-n", "--n_jobs", default=30,
                        required=False, nargs='?')
    # where to save the results/models
    parser.add_argument("-s", "--savepath",
                        default="results_tables", required=False, nargs='?')
    # number of total runs for each experiment
    parser.add_argument("-r", "--num_runs", default=1,
                        required=False, nargs='?')
    # directory containing the task_{id}.csv and task_{id}_categorical_indicator.pkl files
    parser.add_argument("-d", "--data_dir",
                        required=False, nargs='?')
    args = parser.parse_args()
    n_jobs = int(args.n_jobs)
    base_save_folder = args.savepath
    num_runs = int(args.num_runs)
    data_dir = args.data_dir

    save_folder = base_save_folder

    def compute_auroc(model, X_test, y_test):
        y_proba = model.predict_proba(X_test)
        n_classes = len(np.unique(y_test))

        if n_classes == 2:
            return roc_auc_score(y_test, y_proba[:, 1])
        else:
            return roc_auc_score(
                y_test,
                y_proba,
                multi_class="ovr",
                average="macro"
            )


    try:

        task_ids = [
            # binary
            146818, 359955, 168757, 359956, 359958, 359962, 190137,
            168911, 359965, 190411, 146820, 359968, 359975, 359972,
            168350, 359971,
            # multiclass
            190146, 359959, 2073, 359960, 168784, 359963, 359964,
            359974, 359969, 359970,
        ]

        num_runs = 21

        jobs = [(tid, run) for tid in task_ids for run in range(num_runs)]

        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        task_id, run_num = jobs[array_id]

        bagging_search_space = get_bagging_pipeline_space(seed=run_num)

        full_results = []

        print("task id:", task_id, "run num:", run_num)

        # load the data
        data = pd.read_csv(os.path.join(data_dir, f'task_{task_id}.csv'))
        with open(os.path.join(data_dir, f'task_{task_id}_categorical_indicator.pkl'), "rb") as f:
            cat_ind = pickle.load(f)

        data.columns = data.columns.str.strip().str.lower()

        y = data.iloc[:, -1]
        X = data.iloc[:, :-1]

        if len(cat_ind) == data.shape[1]:
            cat_ind = cat_ind[:-1]

        assert len(cat_ind) == X.shape[1]

        cat_cols = X.columns[cat_ind]
        num_cols = X.columns.difference(cat_cols)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2,
            random_state=run_num, stratify=y
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ("num", "passthrough", num_cols),
            ]
        )

        X_train = preprocessor.fit_transform(X_train)
        X_test  = preprocessor.transform(X_test)

        y_train = y_train.to_numpy()
        y_test  = y_test.to_numpy()

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test  = le.transform(y_test)


        # tpot run (50x40) and ES

        est = tpot.TPOTEstimator(search_space=bagging_search_space, generations=50, population_size=40, cv=5, n_jobs=n_jobs, max_time_mins=None,
                                 random_state=run_num, verbose=2, classification=True, scorers=['roc_auc_ovr', tpot.objectives.complexity_scorer], scorers_weights=[1, -1])
        est.fit(X_train, y_train)
        eval_inds = est.evaluated_individuals

        individual_score = compute_auroc(est, X_test, y_test)


        filtered_eval_inds = clean_eval_inds(eval_inds)
        top100 = filtered_eval_inds.nlargest(100, "roc_auc_score")

        # ensemble selection
        # ensemble_random_2000 = greedy_forward_search(filtered_eval_inds, X_train, y_train, run_num)
        ensemble_random_100 = greedy_forward_search(top100, X_train, y_train, run_num)

        # get probas and convert to auroc score
        # ensemble_random_test_proba_2000 = vote_soft_proba(estimators=ensemble_random_2000, X_test=X_test)

        # if len(np.unique(y_test)) == 2:
        #     ensemble_random_test_auroc_2000 = roc_auc_score(
        #         y_test,
        #         ensemble_random_test_proba_2000[:, 1]
        #     )
        # else:
        #     ensemble_random_test_auroc_2000 = roc_auc_score(
        #         y_test,
        #         ensemble_random_test_proba_2000,
        #         multi_class="ovr",
        #         average="macro"
        #     )

        ensemble_random_test_proba_100 = vote_soft_proba(estimators=ensemble_random_100, X_test=X_test)

        if len(np.unique(y_test)) == 2:
            ensemble_random_test_auroc_100 = roc_auc_score(
                y_test,
                ensemble_random_test_proba_100[:, 1]
            )
        else:
            ensemble_random_test_auroc_100 = roc_auc_score(
                y_test,
                ensemble_random_test_proba_100,
                multi_class="ovr",
                average="macro"
            )


        full_results.append({"task id": task_id,
                            "run #": run_num,
                            "individual_tpot": individual_score,
                            # "ensemble_random_2000": ensemble_random_test_auroc_2000,
                            "ensemble_random_100": ensemble_random_test_auroc_100,
                            })

        full_results_df = pd.DataFrame(full_results)
        full_results_df.to_csv(os.path.join(save_folder, (f'bagging_run_{task_id}_#{run_num}.csv')), index=False)

    except Exception as e:
        trace = traceback.format_exc()
        pipeline_failure_dict = {"task_id": task_id,
                                 "run": num_runs, "error": str(e), "trace": trace}
        print("failed on ")
        print(save_folder)
        print(e)
        print(trace)



if __name__ == '__main__':
    main()
    print('DONE')