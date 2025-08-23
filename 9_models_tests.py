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

    # setting values for top 50% and random sampling
    middle_row = pareto_front.shape[0] // 2
    top_half = pareto_front.sort_values(by='roc_auc_score', ascending=False).iloc[:middle_row]

    random_sample = pareto_front.sample(frac=0.5, random_state=seed)

    # evaluates single model performance and creates full estimators list
    for i in range(len(pareto_front)):
        fitted_pipeline = pareto_front.iloc[i, 10].fit(X_train, y_train)

        accuracy = accuracy_score(y_test, fitted_pipeline.predict(X_test))

        if accuracy > highest_accuracy:
            highest_accuracy = accuracy

        voting_weights.append(accuracy)

        fitted_pipeline_tuple = ((str(i), fitted_pipeline))
        estimators.append(fitted_pipeline_tuple)

    # creates top 50% primary objective (auroc) list
    for i in range(len(top_half)):
        fitted_pipeline = top_half.iloc[i, 10].fit(X_train, y_train)
        fitted_pipeline_tuple = ((str(i), fitted_pipeline))
        top_half_estimators.append(fitted_pipeline_tuple)

    # creates random sample list
    for i in range(len(random_sample)):
        fitted_pipeline = random_sample.iloc[i, 10].fit(X_train, y_train)
        fitted_pipeline_tuple = ((str(i), fitted_pipeline))
        random_sample_estimators.append(fitted_pipeline_tuple)

    return estimators, top_half_estimators, random_sample_estimators, voting_weights, highest_accuracy


def vote_hard(estimators, X_test, weights=None):
    # Collect predictions from each estimator
    predictions = np.asarray([est.predict(X_test) for est in estimators]).T  
    
    if weights is None:
        # Majority vote
        return np.array([np.bincount(row).argmax() for row in predictions])
    else:
        # Weighted vote
        weighted_preds = []
        for row in predictions:
            counts = np.bincount(row, weights=weights, minlength=len(np.unique(row)))
            weighted_preds.append(np.argmax(counts))
        return np.array(weighted_preds)




def main():
    parser = argparse.ArgumentParser()
    # number of threads
    parser.add_argument("-n", "--n_jobs", default=30,  required=False, nargs='?')
    #where to save the results/models
    parser.add_argument("-s", "--savepath", default="results_tables", required=False, nargs='?')
    #number of total runs for each experiment
    parser.add_argument("-r", "--num_runs", default=1, required=False, nargs='?')
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


        # load the data
        file_path = f'/common/hodesse/hpc_test/TPOT2_ensemble/data/{task_id}_True.pkl'
        d = pickle.load(open(file_path, "rb"))
        X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']

        est = tpot.TPOTEstimator(search_space=constrained_search_space, generations=100, population_size=50, cv=5, n_jobs=n_jobs, max_time_mins=None,
                                random_state=run_num, verbose=2, classification=True, scorers=['roc_auc_ovr', 'balanced_accuracy'], scorers_weights=[1, 1])
        est.fit(X_train, y_train)
        pf = est.pareto_front

        # save the front
        with open(f'pareto_front_{task_id}_#{run_num}.pkl', "wb") as f:
            pickle.dump(pf, f)

        estimators, top_half_estimators, random_sample_estimators, voting_weights, individual_highest_accuracy = set_up_estimators(
            pf, X_train, y_train, X_test, y_test, run_num)

        # Model 1: includes all, hard voting
        results_1 = vote_hard(estimators=estimators, X_test=X_test)
        accuracy_1 = accuracy_score(y_test, results_1)

        # Model 2: weighted, hard voting
        results_2 = vote_hard(estimators=estimators, X_test=X_test, weights=voting_weights)
        accuracy_2 = accuracy_score(y_test, results_2)



        full_results.append({"task id": task_id,
                            "run #": run_num,
                            "individual": individual_highest_accuracy,
                            "model 1": accuracy_1,
                            "model 2": accuracy_2
                            })

        full_results_df = pd.DataFrame(full_results)
        full_results_df.to_csv(os.path.join(save_folder, f"results_ensemble_{task_id}_#{run_num}.csv"), index=False)



    except Exception as e:
                    trace =  traceback.format_exc()
                    pipeline_failure_dict = {"task_id": task_id,"run": num_runs, "error": str(e), "trace": trace}
                    print("failed on ")
                    print(save_folder)
                    print(e)
                    print(trace)



if __name__ == '__main__':
    main()
    print('DONE')
