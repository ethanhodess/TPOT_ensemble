import openml
import tpot
import sklearn
import traceback
import dill as pickle
import os
import random
import numpy as np
from tpot.search_spaces.pipelines import ChoicePipeline, SequentialPipeline
from estimator_node_gradual import EstimatorNodeGradual
import pandas as pd
import argparse
import ray

from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.metrics import (roc_auc_score, accuracy_score)
from sklearn.base import clone

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

    
@ray.remote
def _ray_get_preds(i, filtered_eval_inds, X_train, y_train, seed):
    # run full CV and return OOF predictions (for clustering pruning)
    est = filtered_eval_inds.iloc[i, 10]
    oof_preds = get_cv_predictions(
        est, X_train, y_train, cv_splits=3, random_state=seed + 105
    )
    return oof_preds


@ray.remote
def _ray_get_probas(estimator, X_train, y_train, cv_splits, random_state):
    # run full CV probas (for ensemble step)
    return get_cv_probas(estimator, X_train, y_train, cv_splits, random_state)


def clean_eval_inds(eval_inds):
    # filter out the broken pipelines
    filtered_eval_inds = eval_inds[eval_inds["roc_auc_score"].notna()]
    print("length of filtered eval_inds:", len(filtered_eval_inds))
    return filtered_eval_inds

# def get_best_individual(eval_inds, X_train, y_train, seed):
#     # get the best individual by auroc from the front
#     pareto_front = eval_inds[eval_inds['Pareto_Front']==1]
#     best_individual = pareto_front.loc[pareto_front["roc_auc_score"].idxmax()]
#     best_individual_est = best_individual.iloc[10]

#     best_individual_preds = get_cv_predictions(
#         best_individual_est, X_train, y_train, cv_splits=5, random_state=seed
#     )
#     best_individual_acc = accuracy_score(y_train, best_individual_preds)

#     return best_individual_est, best_individual_acc

def clustering_pruning(filtered_eval_inds, X_train, y_train, seed):
    # number of clusters
    k = 600

    X_ref = ray.put(X_train)
    y_ref = ray.put(y_train)
    df_ref = ray.put(filtered_eval_inds)

    futures = [
        _ray_get_preds.remote(i, df_ref, X_ref, y_ref, seed) for i in range(len(filtered_eval_inds))
    ]
    
    results = ray.get(futures)
    preds_matrix = np.array(results)

    kmeans = KMeans(n_clusters=k, random_state=seed+210, n_init="auto")
    labels = kmeans.fit_predict(preds_matrix)

    # pick one per cluster
    cluster_chosen = []
    for c in range(k):
        cluster_idx = np.where(labels == c)[0]
        if len(cluster_idx) == 0:
            continue
        cluster_models = filtered_eval_inds.iloc[cluster_idx]
        best_in_cluster = cluster_models.loc[cluster_models["roc_auc_score"].idxmax()]
        cluster_chosen.append(best_in_cluster)

    return pd.DataFrame(cluster_chosen)

def get_ensemble_probas_parallel(estimators, X_train, y_train, cv_splits, random_state):

    X_ref = ray.put(X_train)
    y_ref = ray.put(y_train)

    futures = [
        _ray_get_probas.remote(est, X_ref, y_ref, cv_splits, random_state) for est in estimators
    ]
    return ray.get(futures)

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

    ray.init()

    try:

        task_ids = [359954, 2073, 190146, 168784, 359959]
        num_runs = 21

        jobs = [(tid, run) for tid in task_ids for run in range(num_runs)]

        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        task_id, run_num = jobs[array_id]

        full_results = []
        constrained_search_space = get_pipeline_space(seed=run_num)

        eval_inds_file = f'/common/hodesse/hpc_test/TPOT2_ensemble/short_40_25_eval_inds/complexity_evaluated_individuals_{task_id}_#{run_num}.pkl'

        print("task id:", task_id, "run num:", run_num)

        # load the data
        file_path = (f'/common/hodesse/hpc_test/TPOT2_ensemble/data/{task_id}_True.pkl')
        d = pickle.load(open(file_path, "rb"))
        X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']

        # tpot runs and save evaluated individuals
        if os.path.exists(eval_inds_file):
            with open(eval_inds_file, "rb") as f:
                eval_inds = pickle.load(f)
            est = None
        else:
            est = tpot.TPOTEstimator(search_space=constrained_search_space, generations=40, population_size=25, cv=5, n_jobs=n_jobs, max_time_mins=None,
                                     random_state=run_num, verbose=2, classification=True, scorers=['roc_auc_ovr', tpot.objectives.complexity_scorer], scorers_weights=[1, -1])
            est.fit(X_train, y_train)
            eval_inds = est.evaluated_individuals
            
            # save the evaluated individuals
            with open((f'complexity_evaluated_individuals_{task_id}_#{run_num}.pkl'), "wb") as f:
                pickle.dump(eval_inds, f)

        filtered_eval_inds = clean_eval_inds(eval_inds)
        ensemble = clustering_pruning(filtered_eval_inds, X_train, y_train, run_num)

        tpot_test_accuracy = accuracy_score(y_test, est.predict(X_test)) if est is not None else None

        ensemble_estimators = ensemble.iloc[:, 10].tolist()

        # refit each estimator on full training data
        fitted_ensemble = []
        for est in ensemble_estimators:
            try:
                est_clone = clone(est)
                est_clone.fit(X_train, y_train)
                fitted_ensemble.append(est_clone)
            except Exception as e:
                print(f"estimator failed to fit: {e}")

        ensemble_test_results = vote_soft(estimators=fitted_ensemble, X_test=X_test)
        ensemble_test_accuracy = accuracy_score(y_test, ensemble_test_results)

        full_results.append({"task id": task_id,
                             "run #": run_num,
                             "individual": tpot_test_accuracy,
                             "ensemble": ensemble_test_accuracy
                             })

        full_results_df = pd.DataFrame(full_results)
        full_results_df.to_csv(os.path.join(save_folder, (f'cluster_ensemble_results_{task_id}_#{run_num}.csv')), index=False)

    except Exception as e:
        trace = traceback.format_exc()
        pipeline_failure_dict = {"task_id": task_id,
                                 "run": num_runs, "error": str(e), "trace": trace}
        print("failed on ")
        print(save_folder)
        print(e)
        print(trace)

    ray.shutdown()


if __name__ == '__main__':
    main()
    print('DONE')

