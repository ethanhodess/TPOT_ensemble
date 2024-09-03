import openml
import tpot2
import sklearn.metrics
import sklearn
from sklearn.metrics import (roc_auc_score, log_loss)
import traceback
import dill as pickle
import os
import time
import numpy as np
import sklearn.model_selection
import pandas as pd

from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



def load_task(task_id, preprocess=True):

    cached_data_path = f"data/{task_id}_{preprocess}.pkl"
    print(cached_data_path)
    if os.path.exists(cached_data_path):
        d = pickle.load(open(cached_data_path, "rb"))
        X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']
    else:
        task = openml.tasks.get_task(task_id)


        X, y = task.get_X_and_y(dataset_format="dataframe")
        train_indices, test_indices = task.get_train_test_split_indices()
        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]

        if preprocess:
            preprocessing_pipeline = sklearn.pipeline.make_pipeline(tpot2.builtin_modules.ColumnSimpleImputer("categorical", strategy='most_frequent'), tpot2.builtin_modules.ColumnSimpleImputer("numeric", strategy='mean'), tpot2.builtin_modules.ColumnOneHotEncoder("categorical", min_frequency=0.001, handle_unknown="ignore"))
            X_train = preprocessing_pipeline.fit_transform(X_train)
            X_test = preprocessing_pipeline.transform(X_test)


            le = sklearn.preprocessing.LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

            if task_id == 168795: #this task does not have enough instances of two classes for 10 fold CV. This function samples the data to make sure we have at least 10 instances of each class
                indices = [28535, 28535, 24187, 18736,  2781]
                y_train = np.append(y_train, y_train[indices])
                X_train = np.append(X_train, X_train[indices], axis=0)

            d = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
            if not os.path.exists("data"):
                os.makedirs("data")
            with open(cached_data_path, "wb") as f:
                pickle.dump(d, f)

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = load_task(167104, preprocess=True)


def create_stacking_clf_gridsearch(pareto_front):
    estimators = []
    highest_accuracy = 0
    for i in range(len(pareto_front)):
        fitted_pipeline = pareto_front.iloc[i, 10].fit(X_train, y_train)
        pipeline_accuracy = accuracy_score(y_test, fitted_pipeline.predict(X_test))

        if pipeline_accuracy > highest_accuracy:
            highest_accuracy = pipeline_accuracy
        
        fitted_pipeline_tuple = ((str(i), fitted_pipeline))
        estimators.append(fitted_pipeline_tuple)

    ensemble_model_params = {
        'passthrough': [True, False],  
        'final_estimator': [
            VotingClassifier(estimators=estimators, voting='soft'), 
            VotingClassifier(estimators=estimators, voting='hard'),
            #LogisticRegression()
        ]
    }

    ensemble_grid_search = GridSearchCV(estimator=StackingClassifier(estimators=estimators, cv="prefit"), 
                                        param_grid=ensemble_model_params, 
                                        cv=5)


    ensemble_grid_search.fit(X_train, y_train)
    
    return ensemble_grid_search, highest_accuracy


import argparse

def main():
    parser = argparse.ArgumentParser()
    # number of threads
    parser.add_argument("-n", "--n_jobs", default=30,  required=False, nargs='?')
    #where to save the results/models
    parser.add_argument("-s", "--savepath", default="binary_results", required=False, nargs='?')
    #number of total runs for each experiment
    parser.add_argument("-r", "--num_runs", default=1, required=False, nargs='?')
    args = parser.parse_args()
    n_jobs = int(args.n_jobs)
    base_save_folder = args.savepath
    num_runs = int(args.num_runs)

    save_folder = base_save_folder


    try:

        task_ids = [167104, 167184, 167168, 167161, 189905]
        results = []
        num_runs = 1

        for task_id in task_ids:
            for i in range(num_runs):
                X_train, y_train, X_test, y_test = load_task(task_id, preprocess=True)
                individual_highest_accuracy = 0

                graph_search_space = tpot2.search_spaces.pipelines.GraphPipeline(
                    root_search_space= tpot2.config.get_search_space("classifiers"),
                    leaf_search_space = tpot2.config.get_search_space("selectors"), 
                    inner_search_space = tpot2.config.get_search_space("transformers"),
                    max_size = 10,
                )

                est = tpot2.TPOTEstimator(search_space = graph_search_space, generations=5, population_size=5, cv=5, 
                                    random_state=30+i, verbose=2, classification=True, scorers=['roc_auc_ovr',tpot2.objectives.complexity_scorer], 
                                    scorers_weights=[1,-1])


                est.fit(X_train, y_train)
                fitted_ensemble, individual_highest_accuracy = create_stacking_clf_gridsearch(est.pareto_front)
            
                ensemble_accuracy = accuracy_score(y_test, fitted_ensemble.predict(X_test))
            
                results.append({"task id": task_id, 
                                "individual": individual_highest_accuracy, 
                                "ensemble": ensemble_accuracy, 
                                "better": (ensemble_accuracy>individual_highest_accuracy),
                                "random seed": 30+i,
                                "run #": i
                            })

        print("With grid search")
        results_df = pd.DataFrame(results)
        print(results_df)

        num_better = results_df['better'].sum()
        print("Percent that are better", num_better / results_df['better'].count())


    except Exception as e:
                    trace =  traceback.format_exc()
                    pipeline_failure_dict = {"task_id": task_id,"run": num_runs, "error": str(e), "trace": trace}
                    print("failed on ")
                    print(save_folder)
                    print(e)
                    print(trace)
                    #with open(f"{save_folder}/failed.pkl", "wb") as f:
                    #    pickle.dump(pipeline_failure_dict, f)


if __name__ == '__main__':
    main()
    print('DONE')