import openml
import tpot
import sklearn.metrics
import sklearn
from sklearn.metrics import (roc_auc_score, log_loss)
import traceback
import dill as pickle
import os
import time
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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.base import clone
import argparse


import warnings
warnings.filterwarnings("ignore")

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
        est_clone.fit(X_train[train_idx], y_train[train_idx])
        cv_preds[valid_idx] = est_clone.predict(X_train[valid_idx])

    return cv_preds

def get_cv_probas(estimator, X_train, y_train, cv_splits, random_state):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    n_classes = len(np.unique(y_train))
    cv_probas = np.zeros((len(y_train), n_classes))

    for train_idx, valid_idx in cv.split(X_train, y_train):
        est_clone = clone(estimator)
        est_clone.fit(X_train[train_idx], y_train[train_idx])
        cv_probas[valid_idx] = est_clone.predict_proba(X_train[valid_idx])

    return cv_probas

def set_up_estimators(eval_inds, pareto_front, X_train, y_train, X_test, y_test, seed):
    
    highest_accuracy = 0
    
    # evaluates single model performance on test set
    for i in range(len(pareto_front)):
        fitted_pipeline = pareto_front.iloc[i, 10].fit(X_train, y_train)

        accuracy = accuracy_score(y_test, fitted_pipeline.predict(X_test))

        if accuracy > highest_accuracy:
            highest_accuracy = accuracy

        
    # filter estimators to include top 100 auroc and top 100 bal acc
    top_auroc = eval_inds.nlargest(100, "roc_auc_score")
    remaining = eval_inds.drop(top_auroc.index)
    top_bal_acc = remaining.nlargest(100, "balanced_accuracy_score")
    top_200 = pd.concat([top_auroc, top_bal_acc])

    # pull the pipelines and fit 
    filtered_estimators = []
    for i in range(len(top_200)):
        filtered_estimators.append(top_200.iloc[i, 10].fit(X_train, y_train))

    # creates diverse pipeline list
    diverse_estimators = []
    voting_weights_diverse = []

    # cache CV predictions, accuracies
    est_cv_preds = {}
    est_cv_probas = {}
    est_cv_acc = {}

    for est in filtered_estimators:
        est_cv_probas[est] = get_cv_probas(est, X_train, y_train, cv_splits=5, random_state=seed)

        preds = get_cv_predictions(estimator=est,
                                   X_train=X_train, y_train=y_train,
                                   cv_splits=5, random_state=seed)
        acc = accuracy_score(y_train, preds)
        est_cv_preds[est] = preds
        est_cv_acc[est] = acc

    # select best estimator by CV accuracy score
    best_estimator_cv = max(filtered_estimators, key=lambda e: est_cv_acc[e])
    best_acc_cv = est_cv_acc[best_estimator_cv]

    # start with best estimator
    diverse_estimators.append(best_estimator_cv)
    voting_weights_diverse.append(best_acc_cv)
    ensemble_preds = est_cv_preds[best_estimator_cv]
    ensemble_acc = est_cv_acc[best_estimator_cv]


    # greedy selection
    while True:
        improvement_found = False
        best_candidate = None
        best_candidate_acc = ensemble_acc
        best_candidate_preds = None
        best_candidate_weight = None

        for est in filtered_estimators:
            if est in diverse_estimators:
                continue

            # test ensemble CV accuracy when each candidate is added
            candidate_probas = [est_cv_probas[e] for e in diverse_estimators + [est]]
            candidate_weights = voting_weights_diverse + [est_cv_acc[est]]
            temp_preds = combine_preds(candidate_probas, candidate_weights)
            temp_acc = accuracy_score(y_train, temp_preds)

            if temp_acc > best_candidate_acc:
                best_candidate = est
                best_candidate_acc = temp_acc
                best_candidate_preds = temp_preds
                best_candidate_weight = est_cv_acc[est]
            
        if best_candidate is not None:
            print(f"Adding model with CV acc {est_cv_acc[best_candidate]:.4f}, "
                  f"ensemble acc improves to {best_candidate_acc:.4f}")
            diverse_estimators.append(best_candidate)
            voting_weights_diverse.append(best_candidate_weight)
            ensemble_preds = best_candidate_preds
            ensemble_acc = best_candidate_acc
            improvement_found = True
        
        if not improvement_found:
            break

    return highest_accuracy, diverse_estimators, voting_weights_diverse


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

        task_ids = [359954, 2073, 190146, 168784, 359959]
        num_runs = 15

        jobs = [(tid, run) for tid in task_ids for run in range(num_runs)]

        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        task_id, run_num = jobs[array_id]

        full_results = []
        constrained_search_space = get_pipeline_space(seed=run_num)

        eval_inds_file = f'/common/hodesse/hpc_test/TPOT2_ensemble/saved_eval_inds/evaluated_individuals_{task_id}_#{run_num}.pkl'
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
                                     random_state=run_num, verbose=2, classification=True, scorers=['roc_auc_ovr', 'balanced_accuracy'], scorers_weights=[1, 1])
            est.fit(X_train, y_train)
            eval_inds = est.evaluated_individuals
            

            # save the front
            with open((f'evaluated_individuals_{task_id}_#{run_num}.pkl'), "wb") as f:
                pickle.dump(eval_inds, f)

        individual_highest_accuracy, diverse_estimators, voting_weights_diverse = set_up_estimators(
            eval_inds, pf, X_train, y_train, X_test, y_test, run_num)

        # Model 1: diverse, hard voting, 
        results_1 = vote_hard(estimators=diverse_estimators, X_test=X_test, weights=voting_weights_diverse)
        accuracy_1 = accuracy_score(y_test, results_1)

        # Model 2: diverse, soft voting, 
        results_2 = vote_soft(estimators=diverse_estimators, X_test=X_test, weights=voting_weights_diverse)
        accuracy_2 = accuracy_score(y_test, results_2)

        full_results.append({"task id": task_id,
                             "run #": run_num,
                             "individual": individual_highest_accuracy,
                             "model 1": accuracy_1,
                             "model 2": accuracy_2
                             })

        full_results_df = pd.DataFrame(full_results)
        full_results_df.to_csv(os.path.join(save_folder, (f'results_ensemble_{task_id}_#{run_num}.csv')), index=False)

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
