import openml
import tpot
import sklearn.metrics
import sklearn
from sklearn.metrics import (roc_auc_score, log_loss)
import traceback
import dill as pickle
import os
import time
import random
import numpy as np
import sklearn.model_selection
from tpot.search_spaces.pipelines import ChoicePipeline, SequentialPipeline
from functools import partial
from estimator_node_gradual import EstimatorNodeGradual
import pandas as pd

from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.base import clone
import argparse
from joblib import Parallel, delayed


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


def get_cv_predictions(estimator, X_train, y_train, cv_splits, random_state):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    cv_preds = np.empty(len(y_train), dtype=int)

    for train_idx, valid_idx in cv.split(X_train, y_train):
        est_clone = clone(estimator) 

        try:
            est_clone.fit(X_train[train_idx], y_train[train_idx])
            cv_preds[valid_idx] = est_clone.predict(X_train[valid_idx])
        except Exception as E:
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
        except Exception as E:
            print('pipeline failed')       
    return cv_probas


def set_up_estimators(eval_inds, X_train, y_train, seed):
    
    # filter out the broken pipelines
    filtered_eval_inds = eval_inds[eval_inds["roc_auc_score"].notna()]

    # randomly sample 500 pipelines (100 from each acc quantile)
    filtered_eval_inds["quantile"] = pd.qcut(filtered_eval_inds["roc_auc_score"], q=5, labels=False)
    mid_pool = filtered_eval_inds.groupby("quantile").apply(
        lambda x: x.sample(n=min(100, len(x)), random_state=seed)
    ).reset_index(drop=True)
        
    
    # prediction clustering for diversity   
    # sample indices for clustering (stratified by class)
    sample_size = 50
    rng = np.random.RandomState(seed + 40)

    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    class_proportions = class_counts / len(y_train)
    samples_per_class = np.maximum(1, np.round(class_proportions * sample_size).astype(int))

    # rounding adjustment
    diff = sample_size - samples_per_class.sum()
    if diff != 0:
        # add/remove from the largest class
        largest_class_idx = np.argmax(samples_per_class)
        samples_per_class[largest_class_idx] += diff

    # save indices
    sample_idx = []
    for cls, n_samples in zip(unique_classes, samples_per_class):
        cls_indices = np.where(y_train == cls)[0]
        chosen = rng.choice(cls_indices, size=n_samples, replace=False)
        sample_idx.append(chosen)

    sample_idx = np.concatenate(sample_idx)

    # cache the oof predictions
    def _get_preds(i):
        est = mid_pool.iloc[i, 10]
        X_sub = X_train[sample_idx]
        y_sub = y_train[sample_idx]
        oof_preds = get_cv_predictions(est, X_sub, y_sub, cv_splits=3, random_state=seed+20)

        return oof_preds, est
    
    print("start cache preds")
    results = Parallel(n_jobs=-1)(
        delayed(_get_preds)(i) for i in range(len(mid_pool))
    )
    preds_matrix_list, est_list = zip(*results)
    preds_matrix = np.array(preds_matrix_list)

    # clustering
    print("clustering start")
    k = 60
    kmeans = KMeans(n_clusters=k, random_state=seed+40, n_init="auto")
    labels = kmeans.fit_predict(preds_matrix)

    # pick one per cluster
    cluster_chosen = []
    for c in range(k):
        cluster_idx = np.where(labels == c)[0]
        if len(cluster_idx) == 0:
            continue
        cluster_models = mid_pool.iloc[cluster_idx]
        best_in_cluster = cluster_models.loc[cluster_models["roc_auc_score"].idxmax()]
        cluster_chosen.append(best_in_cluster)

    cluster_df = pd.DataFrame(cluster_chosen)

    print("clustering finished")

    # pull the pipelines and fit 
    filtered_candidates = []
    for i in range(len(cluster_df)):
        try:
            filtered_candidates.append(cluster_df.iloc[i, 10].fit(X_train, y_train))
        except Exception as E:
            print('fit failed')

    # creates diverse pipeline list
    diverse_estimators_running = []
    best_ensemble = []
    best_ensemble_acc = 0

    # cache CV predictions, accuracies
    est_cv_preds = {}
    est_cv_probas = {}
    est_cv_acc = {}

    for est in filtered_candidates:
        est_cv_probas[est] = get_cv_probas(est, X_train, y_train, cv_splits=5, random_state=seed+60)

        preds = get_cv_predictions(estimator=est,
                                   X_train=X_train, y_train=y_train,
                                   cv_splits=5, random_state=seed+60)
        acc = accuracy_score(y_train, preds)
        est_cv_preds[est] = preds
        est_cv_acc[est] = acc

    # initialize with top 5 
    sorted_estimators = sorted(filtered_candidates, key=lambda e: est_cv_acc[e], reverse=True)
    top5_estimators = sorted_estimators[:5]
    diverse_estimators_running.extend(top5_estimators)
    #top5_acc = [est_cv_acc[e] for e in top5_estimators]

    print("greedy selection start")
    # greedy selection
    for i in range(25):
        best_candidate = None
        best_candidate_acc = 0

        # bagged selection (50% of candidates eligible each step)
        subset = random.sample(filtered_candidates, k=len(filtered_candidates)//2)
        for est in subset:

            # test ensemble CV accuracy when each candidate is added
            candidate_probas = [est_cv_probas[e] for e in diverse_estimators_running + [est]]
            temp_preds = combine_preds(candidate_probas)
            temp_acc = accuracy_score(y_train, temp_preds)

            if temp_acc > best_candidate_acc:
                best_candidate = est
                best_candidate_acc = temp_acc
        
        print(f"ensemble acc changes to {best_candidate_acc:.4f}")
        diverse_estimators_running.append(best_candidate)
        temp_ensemble_probas = [est_cv_probas[e] for e in diverse_estimators_running]
        temp_ensemble_preds = combine_preds(temp_ensemble_probas)
        temp_ensemble_acc = accuracy_score(y_train, temp_ensemble_preds)

        if(temp_ensemble_acc > best_ensemble_acc):
            best_ensemble_acc = temp_ensemble_acc
            best_ensemble = diverse_estimators_running.copy()

    print(f"FINAL best ensemble acc: {best_ensemble_acc:.4f}")
    print(f"FINAL ensemble size: {len(best_ensemble)}")
    return best_ensemble


def vote_hard(estimators, X_test, weights=None):
    predictions = np.asarray([est.predict(X_test) for est in estimators]).T
    if weights is None:
        return np.array([np.bincount(row).argmax() for row in predictions])
    else:
        weighted_preds = []
        for row in predictions:
            counts = np.bincount(row, weights=weights,
                                 minlength=len(np.unique(row)))
            weighted_preds.append(np.argmax(counts))
        return np.array(weighted_preds)
    
def vote_soft(estimators, X_test, weights=None):
    probas = np.stack([est.predict_proba(X_test) for est in estimators])
    if weights is not None:
        weights = np.asarray(weights).reshape(-1, 1, 1)
        probas *= weights
    
    return np.argmax(probas.sum(axis=0), axis=1)

def combine_preds(proba_list, weights=None):
    probas = np.stack(proba_list, axis=0)

    if weights is not None:
        weights = np.asarray(weights).reshape(-1, 1, 1)  # (n_models, 1, 1)
        probas = probas * weights

    # Average (or weighted sum) across models
    avg_proba = probas.sum(axis=0) / (weights.sum() if weights is not None else len(proba_list))

    # Pick the class with max probability
    return np.argmax(avg_proba, axis=1)



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
    args = parser.parse_args()
    n_jobs = int(args.n_jobs)
    base_save_folder = args.savepath
    num_runs = int(args.num_runs)

    save_folder = base_save_folder

    try:

        #task_ids = [359954, 2073, 190146, 168784, 359959]
        task_ids = [359959]
        num_runs = 3

        jobs = [(tid, run) for tid in task_ids for run in range(num_runs)]

        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        task_id, run_num = jobs[array_id]

        full_results = []
        constrained_search_space = get_pipeline_space(seed=run_num)

        eval_inds_file = f'/common/hodesse/hpc_test/TPOT2_ensemble/test_eval_inds/evaluated_individuals_{task_id}_#{run_num}.pkl'
        pf_file = f'/common/hodesse/hpc_test/TPOT2_ensemble/saved_fronts/pareto_front_{task_id}_#{run_num}.pkl'

        print("task id:", task_id, "run num:", run_num)

        # load the data
        file_path = (f'/common/hodesse/hpc_test/TPOT2_ensemble/data/{task_id}_True.pkl')
        d = pickle.load(open(file_path, "rb"))
        X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']

        # loads the pareto front
        if os.path.exists(pf_file):
            with open(pf_file, "rb") as f:
                pf = pickle.load(f)

        # tpot runs and save evaluated individuals
        if os.path.exists(eval_inds_file):
            with open(eval_inds_file, "rb") as f:
                eval_inds = pickle.load(f)
        else:
            est = tpot.TPOTEstimator(search_space=constrained_search_space, generations=100, population_size=50, cv=5, n_jobs=n_jobs, max_time_mins=None,
                                     random_state=run_num, verbose=2, classification=True, scorers=['roc_auc_ovr', tpot.objectives.complexity_scorer], scorers_weights=[1, -1])
            est.fit(X_train, y_train)
            eval_inds = est.evaluated_individuals
            

            # save the front
            with open((f'evaluated_individuals_{task_id}_#{run_num}.pkl'), "wb") as f:
                pickle.dump(eval_inds, f)

        best_ensemble = set_up_estimators(
            eval_inds, X_train, y_train, run_num)
        
        #tpot_accuracy = accuracy_score(y_test, est.predict(X_test))

        # Model 2: diverse, soft voting, 
        results_2 = vote_soft(estimators=best_ensemble, X_test=X_test)
        accuracy_2 = accuracy_score(y_test, results_2)

        full_results.append({"task id": task_id,
                             "run #": run_num,
                             #"individual": tpot_accuracy,
                             "model 2": accuracy_2
                             })

        full_results_df = pd.DataFrame(full_results)
        full_results_df.to_csv(os.path.join(save_folder, (f'test_APR_results_ensemble_{task_id}_#{run_num}.csv')), index=False)

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
