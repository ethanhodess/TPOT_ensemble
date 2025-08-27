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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.base import clone
import argparse


# defines a constrained search space with only three steps

def get_pipeline_space(seed):
    return tpot.search_spaces.pipelines.SequentialPipeline([
        tpot.config.get_search_space(
            ["selectors_classification", "Passthrough"], random_state=seed, base_node=EstimatorNodeGradual),
        tpot.config.get_search_space(
            ["transformers", "Passthrough"], random_state=seed, base_node=EstimatorNodeGradual),
        tpot.config.get_search_space("classifiers", random_state=seed, base_node=EstimatorNodeGradual)])


def set_up_estimators(pareto_front, X_train, y_train, X_test, y_test, seed):
    estimators = []
    voting_weights = []
    top_half_estimators = []
    random_sample_estimators = []
    highest_accuracy = 0
    best_estimator = None
    diverse_estimators = []
    corr_threshold = 0.90
    voting_weights_diverse = []

    # setting values for top 50% and random sampling
    middle_row = pareto_front.shape[0] // 2
    top_half = pareto_front.sort_values(by='balanced_accuracy_score', ascending=False).iloc[:middle_row]

    random_sample = pareto_front.sample(frac=0.5, random_state=seed)

    # evaluates single model performance and creates full estimators list
    for i in range(len(pareto_front)):
        fitted_pipeline = pareto_front.iloc[i, 10].fit(X_train, y_train)

        accuracy = accuracy_score(y_test, fitted_pipeline.predict(X_test))

        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_estimator = fitted_pipeline

        voting_weights.append(accuracy**40)

        #fitted_pipeline_tuple = ((str(i), fitted_pipeline))
        estimators.append(fitted_pipeline)

    # creates top 50% primary objective (bacc) list
    for i in range(len(top_half)):
        fitted_pipeline = top_half.iloc[i, 10].fit(X_train, y_train)
        #fitted_pipeline_tuple = ((str(i), fitted_pipeline))
        top_half_estimators.append(fitted_pipeline)

    # creates random sample list
    for i in range(len(random_sample)):
        fitted_pipeline = random_sample.iloc[i, 10].fit(X_train, y_train)
        fitted_pipeline_tuple = ((str(i), fitted_pipeline))
        random_sample_estimators.append(fitted_pipeline_tuple)

    # creates diverse pipeline list
    if best_estimator is not None:
        best_preds = best_estimator.predict(X_test)
        diverse_estimators.append(best_estimator)
        voting_weights_diverse.append(highest_accuracy**40)

        for est in estimators:
            preds = est.predict(X_test)
            if np.std(preds) == 0 or np.std(best_preds) == 0:  # avoid NaN corr
                corr = 1.0 if np.all(preds == best_preds) else 0.0
            else:
                corr = np.corrcoef(best_preds, preds)[0, 1]

            if corr < corr_threshold:
                diverse_estimators.append(est)
                acc = accuracy_score(y_test, preds)
                voting_weights_diverse.append(acc**40)

    return estimators, top_half_estimators, random_sample_estimators, voting_weights, highest_accuracy, diverse_estimators, voting_weights_diverse


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

        pf_file = f'/common/hodesse/hpc_test/TPOT2_ensemble/saved_fronts/pareto_front_{task_id}_#{run_num}.pkl'

        # load the data
        file_path = (f'/common/hodesse/hpc_test/TPOT2_ensemble/data/{task_id}_True.pkl')
        d = pickle.load(open(file_path, "rb"))
        X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']

        # tpot runs and save fronts
        if os.path.exists(pf_file):
            with open(pf_file, "rb") as f:
                pf = pickle.load(f)
        else:
            est = tpot.TPOTEstimator(search_space=constrained_search_space, generations=100, population_size=50, cv=5, n_jobs=n_jobs, max_time_mins=None,
                                     random_state=run_num, verbose=2, classification=True, scorers=['roc_auc_ovr', 'balanced_accuracy'], scorers_weights=[1, 1])
            est.fit(X_train, y_train)
            pf = est.pareto_front

            # save the front
            with open((f'pareto_front_{task_id}_#{run_num}.pkl'), "wb") as f:
                pickle.dump(pf, f)

        estimators, top_half_estimators, random_sample_estimators, voting_weights, individual_highest_accuracy, diverse_estimators, voting_weights_diverse = set_up_estimators(
            pf, X_train, y_train, X_test, y_test, run_num)

        # Model 1: all, hard voting, weighted
        results_1 = vote_hard(estimators=diverse_estimators, X_test=X_test, weights=voting_weights_diverse)
        accuracy_1 = accuracy_score(y_test, results_1)

        # Model 2: all, soft voting, weighted
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
