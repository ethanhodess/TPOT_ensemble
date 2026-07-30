# Python script to pull datasets from OpenML benchmark suite 271


import os
import csv
import argparse
import sys
import pickle
from openml import tasks
from openml.study import get_suite
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# task_ids = [2073, 3945, 7593, 10090, 146818, 146820, 167120, 168350, 
#             168757, 168784, 168868, 168909, 168910, 168911, 189354, 189355, 
#             189356, 189922, 190137, 190146, 190392, 190410, 190411, 190412, 
#             211979, 211986, 359953, 359954, 359955, 359956, 359957, 359958, 
#             359959, 359960, 359961, 359962, 359963, 359964, 359965, 359966, 
#             359967, 359968, 359969, 359970, 359971, 359972, 359973, 359974, 
#             359975, 359976, 359977, 359979, 359980, 359981, 359982, 359983, 
#             359984, 359985, 359986, 359987, 359988, 359989, 359990, 359991, 
#             359992, 359993, 359994, 360112, 360113, 360114, 360975]

task_ids = [146818, 359955, 190146, 168757, 359956]

OUTPUT_DIR = "data"
 
 
def load_task_dataset(task):
    dataset = task.get_dataset()
    target_name = task.target_name
 
    X, y, _, _ = dataset.get_data(
        target=target_name,
        dataset_format="dataframe"
    )
 
    df = X.copy()
    df[target_name] = y
    has_missing = df.isna().any().any()
 
    class_counts = pd.Series(y).value_counts()
    total_count = len(y)
    minority_pct = (class_counts.min() / total_count) * 100
    majority_pct = (class_counts.max() / total_count) * 100
 
    return df, target_name, has_missing, minority_pct, majority_pct
 
 
def process_task(task_id):
    print(f"Processing task {task_id}...")
    task = tasks.get_task(task_id)
 
    df, target_name, has_missing, minority_pct, majority_pct = load_task_dataset(task)
 
    if has_missing:
        print(f"  -> Task {task_id} has missing values. Skipping.")
        return
 
    n_rows, n_cols = df.shape
 
    dataset = task.get_dataset()
    _, _, categorical_indicator, _ = dataset.get_data(
        target=target_name,
        dataset_format="dataframe"
    )
 
    cat_path = os.path.join(OUTPUT_DIR, f"task_{task_id}_categorical_indicator.pkl")
    with open(cat_path, "wb") as f:
        pickle.dump(categorical_indicator, f)
 
    csv_path = os.path.join(OUTPUT_DIR, f"task_{task_id}.csv")
    df.to_csv(csv_path, index=False)
 
    print(f"  -> Saved dataset to {csv_path}")
    print(f"  -> Saved categorical indicator to {cat_path}")
    print(f"  -> Rows: {n_rows}, Columns: {n_cols}, Minority: {minority_pct:.2f}%, Majority: {majority_pct:.2f}%")
 
 
if __name__ == "__main__": 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}") 
    for task_id in task_ids:
        try:
            process_task(task_id)
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")