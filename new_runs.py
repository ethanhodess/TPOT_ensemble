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

        constrained_search_space = get_pipeline_space(seed=run_num)


        print("task id:", task_id, "run num:", run_num)

        # load the data
        file_path = (f'/common/hodesse/hpc_test/TPOT2_ensemble/data/{task_id}_True.pkl')
        d = pickle.load(open(file_path, "rb"))
        X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']



        est = tpot.TPOTEstimator(search_space=constrained_search_space, generations=100, population_size=50, cv=5, n_jobs=n_jobs, max_time_mins=None,
                                 random_state=run_num, verbose=2, classification=True, scorers=['roc_auc_ovr', 'balanced_accuracy'], scorers_weights=[1, 1])
        est.fit(X_train, y_train)
        eval_inds = est.evaluated_individuals

        # save the evaluated individuals
        outpath = os.path.join(save_folder, f"evaluated_individuals_{task_id}_#{run_num}.pkl")
        with open(outpath, "wb") as f:
            pickle.dump(eval_inds, f)


        
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
