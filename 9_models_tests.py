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


def set_up_estimators(pareto_front, X_train, y_train, X_test, y_test):
    estimators = []
    voting_weights = []
    top_half_estimators = []
    random_sample_estimators = []
    highest_accuracy = 0

    # setting values for top 50% and random sampling
    middle_row = pareto_front.shape[0] // 2
    top_half = pareto_front.sort_values(
        by='roc_auc_score', ascending=False).iloc[:middle_row]
    random_sample = pareto_front.sample(frac=0.5, random_state=42)

    # evaluates single model performance and creates full estimators list
    for i in range(len(pareto_front)):
        fitted_pipeline = pareto_front.iloc[i, 10].fit(X_train, y_train)

        accuracy = accuracy_score(y_test, fitted_pipeline.predict(X_test))

        if accuracy > highest_accuracy:
            highest_accuracy = accuracy

        voting_weights.append(accuracy**2)

        fitted_pipeline_tuple = ((str(i), fitted_pipeline))
        estimators.append(fitted_pipeline_tuple)

    # creates top 50% primary objective (accuracy) list
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

        task_ids = [359954, 2073, 190146, 168784, 359959]
        num_runs = 15

        full_results = []
        constrained_search_space = get_pipeline_space(seed=42)


        for task_id in task_ids:
            for i in range(num_runs):

                # load the data
                file_path = f'/data/{task_id}_True.pkl'
                d = pickle.load(open(file_path, "rb"))
                X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']

                # individual_highest_accuracy = 0
                est = tpot.TPOTEstimator(search_space=constrained_search_space, generations=100, population_size=50, cv=5,
                                        random_state=42+i, verbose=2, classification=True, scorers=['roc_auc_ovr', tpot.objectives.complexity_scorer], scorers_weights=[1, -1])
                est.fit(X_train, y_train)
                pf = est.pareto_front

                estimators, top_half_estimators, random_sample_estimators, voting_weights, individual_highest_accuracy = set_up_estimators(
                    pf, X_train, y_train, X_test, y_test)

                # Model 1: includes all, hard voting
                model_1 = VotingClassifier(estimators=estimators, voting='hard')

                model_1.fit(X_train, y_train)
                results = model_1.predict(X_test)
                accuracy_1 = accuracy_score(y_test, results)

                # Model 2: includes all, soft voting
                model_2 = VotingClassifier(estimators=estimators, voting='soft')

                model_2.fit(X_train, y_train)
                results = model_2.predict(X_test)
                accuracy_2 = accuracy_score(y_test, results)

                # Model 3: top 50%, hard voting
                model_3 = VotingClassifier(estimators=top_half_estimators, voting='hard')

                model_3.fit(X_train, y_train)
                results = model_3.predict(X_test)
                accuracy_3 = accuracy_score(y_test, results)

                # Model 4: top 50%, soft voting
                model_4 = VotingClassifier(estimators=top_half_estimators, voting='soft')

                model_4.fit(X_train, y_train)
                results = model_4.predict(X_test)
                accuracy_4 = accuracy_score(y_test, results)

                # Model 5: Random sample, hard voting
                model_5 = VotingClassifier(estimators=random_sample_estimators, voting='hard')

                model_5.fit(X_train, y_train)
                results = model_5.predict(X_test)
                accuracy_5 = accuracy_score(y_test, results)

                # Model 6: Random sample, soft voting
                model_6 = VotingClassifier(estimators=random_sample_estimators, voting='soft')

                model_6.fit(X_train, y_train)
                results = model_6.predict(X_test)
                accuracy_6 = accuracy_score(y_test, results)

                # Model 7: Weighted, hard voting
                model_7 = VotingClassifier(estimators=estimators, voting='hard', weights=voting_weights)

                model_7.fit(X_train, y_train)
                results = model_7.predict(X_test)
                accuracy_7 = accuracy_score(y_test, results)

                # Model 8: Weighted, soft voting
                model_8 = VotingClassifier(estimators=estimators, voting='soft', weights=voting_weights)

                model_8.fit(X_train, y_train)
                results = model_8.predict(X_test)
                accuracy_8 = accuracy_score(y_test, results)


                full_results.append({"task id": task_id,
                                    "run #": i,
                                    "individual": individual_highest_accuracy,
                                    "model 1": accuracy_1,
                                    "model 2": accuracy_2,
                                    "model 3": accuracy_3,
                                    "model 4": accuracy_4,
                                    "model 5": accuracy_5,
                                    "model 6": accuracy_6,
                                    "model 7": accuracy_7,
                                    "model 8": accuracy_8
                                    })

        full_results_df = pd.DataFrame(full_results)
        full_results_df.to_csv(os.path.join(save_folder, f"results_ensemble.csv"), index=False)



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
