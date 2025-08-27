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

        pf_file = f'/common/hodesse/hpc_test/TPOT2_ensemble/saved_fronts/pareto_front_{task_id}_#{run_num}.pkl'
        with open(pf_file, "rb") as f:
            pf = pickle.load(f)

        all_rows = []

        # load the data
        file_path = (f'/common/hodesse/hpc_test/TPOT2_ensemble/data/{task_id}_True.pkl')
        d = pickle.load(open(file_path, "rb"))
        X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']

        accuracies = []
        
        # First pass: compute accuracies only
        for i in range(len(pf)):
            pipe = pf.iloc[i, 10]
            acc = accuracy_score(y_test, pipe.fit(X_train, y_train).predict(X_test))
            accuracies.append(acc)

        best_idx = int(np.argmax(accuracies))
        best_acc = accuracies[best_idx]

        # Refit best pipeline once
        best_pipe = pf.iloc[best_idx, 10].fit(X_train, y_train)
        best_preds = best_pipe.predict(X_test)

        # Second pass: compute correlations vs best
        for i in range(len(pf)):
            pipe = pf.iloc[i, 10]
            preds = pipe.fit(X_train, y_train).predict(X_test)
            corr = np.corrcoef(best_preds, preds)[0, 1] if i != best_idx else 1.0

            all_rows.append({
                "task_id": task_id,
                "run_num": run_num,
                "pipeline_index": i,
                "accuracy": accuracies[i],
                "is_best": (i == best_idx),
                "best_accuracy": best_acc,
                "correlation_with_best": corr
            })

        summary_df = pd.DataFrame(all_rows)
        summary_df.to_csv(f'pipeline_corr_results_summary_{task_id}_{run_num}.csv', index=False)

        

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
