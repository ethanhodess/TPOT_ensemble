# Python script to pull binary classification datasets from OpenML benchmark suite 271
# The script filters for classification tasks and datasets that meet row and column thresholds
# Datasets with missing values are excluded.
# Datasets that meet the criteria are downloaded and saved as CSV files.
# A summary CSV listing the task IDs, number of rows, and number of columns, and number of classes.
# link: https://www.openml.org/search?type=benchmark&study_type=task&sort=tasks_included&id=271
# paper: https://www.jmlr.org/papers/volume25/22-0493/22-0493.pdf

import os
import csv
import argparse
from openml import tasks
from openml.study import get_suite
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typeguard import typechecked

@typechecked
def get_suite_task_ids(suite_id: int = 271):
    """Return the list of task IDs in the given OpenML benchmark suite."""
    suite = get_suite(suite_id)  # OpenMLBenchmarkSuite object
    return suite.tasks  # list of task IDs (ints)

@typechecked
def load_task_dataset(task):
    """
    Load the full dataset for a task into a pandas DataFrame.
    Returns (df, target_name, has_missing, minority_pct, majority_pct).
    """
    dataset = task.get_dataset()
    target_name = task.target_name

    X, y, _, _ = dataset.get_data(
        target=target_name,
        dataset_format="dataframe"
    )

    # Combine X and y into a single DataFrame
    df = X.copy()
    df[target_name] = y
    has_missing = df.isna().any().any()

    # Calculate class percentages
    class_counts = pd.Series(y).value_counts()
    total_count = len(y)
    minority_count = class_counts.min()
    majority_count = class_counts.max()
    minority_pct = (minority_count / total_count) * 100
    majority_pct = (majority_count / total_count) * 100

    # get total number of unique classes
    num_classes = len(class_counts)

    return df, target_name, has_missing, minority_pct, majority_pct, num_classes

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Pull binary classification datasets from OpenML suite 271')
    parser.add_argument('--save-dir', type=str, default='',
                        help='Directory path to append to the default output directory name')
    args = parser.parse_args()

    # Maximum thresholds for filtering datasets (set to None to disable filtering)
    MAX_ROWS_THRESHOLD = 50000
    MAX_COLS_THRESHOLD = 1000

    suite_id = 271
    base_output_dir = "Raw_OpenML_Suite_271_Classification"

    # Append the save-dir argument to the base output directory if provided
    if args.save_dir:
        output_dir = os.path.join(args.save_dir, base_output_dir)
    else:
        output_dir = base_output_dir

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # 1. Get all task IDs from the suite
    task_ids = get_suite_task_ids(suite_id)
    print(f"Found {len(task_ids)} tasks in suite {suite_id}.")

    # FIRST PASS: Filter datasets based on thresholds (before encoding)
    print("\n" + "="*60)
    print("FIRST PASS: Filtering datasets based on thresholds")
    print("="*60)

    filtered_task_ids = []

    # Loop through each task ID and check if it meets the criteria
    for task_id in task_ids:
        try:
            print(f"Checking task {task_id}...")
            task = tasks.get_task(task_id)

            # Load the dataset for this task
            df, target_name, has_missing, minority_pct, majority_pct, num_classes = load_task_dataset(task)
            n_rows, n_cols = df.shape

            # Skip datasets with missing values
            if has_missing:
                print(f"  -> Task {task_id} has missing values. Skipping.")
                continue

            # Check thresholds on raw data (before one-hot encoding)
            if MAX_ROWS_THRESHOLD is not None and n_rows >= MAX_ROWS_THRESHOLD:
                print(f"  -> Task {task_id} exceeds MAX_ROWS_THRESHOLD ({n_rows} >= {MAX_ROWS_THRESHOLD}). Skipping.")
                continue

            if MAX_COLS_THRESHOLD is not None and n_cols >= MAX_COLS_THRESHOLD:
                print(f"  -> Task {task_id} exceeds MAX_COLS_THRESHOLD ({n_cols} >= {MAX_COLS_THRESHOLD}). Skipping.")
                continue

            print(f"  -> Task {task_id} meets criteria. N: {n_rows}, M: {n_cols}, Minority: {minority_pct:.2f}%, Majority: {majority_pct:.2f}%, Num classes: {num_classes}")
            filtered_task_ids.append(task_id)

        except Exception as e:
            print(f"  -> Error checking task {task_id}: {e}")

    print(f"\nFiltered to {len(filtered_task_ids)} out of {len(task_ids)} tasks")
    print("\n" + "="*60)
    print("SECOND PASS: Encoding and saving filtered datasets")
    print("="*60)

    summary_rows = []

    # encode y column to numerical labels starting from 0 and save to CSV
    for task_id in filtered_task_ids:
        try:
            print(f"Processing task {task_id}...")
            task = tasks.get_task(task_id)

            # 2. Load the dataset for this task
            df, target_name, has_missing, minority_pct, majority_pct, num_classes = load_task_dataset(task)

            n_rows, n_cols = df.shape

            # 3. Encode target variable to numerical labels starting from 0
            label_encoder = LabelEncoder()
            encoded_target = label_encoder.fit_transform(df[target_name])
            df = df.drop(columns=[target_name])
            df[target_name] = np.asarray(encoded_target)

            # Save label encoder mapping for reference
            label_mapping = {original: encoded for encoded, original in enumerate(label_encoder.classes_)}
            print(f"  -> Target label mapping: {label_mapping}")

            # 4. Save the processed dataset to CSV
            dataset_csv_path = os.path.join(output_dir, f"task_{task_id}.csv")
            df.to_csv(dataset_csv_path, index=False)

            print(f"  -> Saved to {dataset_csv_path}")
            print(f"  -> Final dimensions - Rows: {n_rows}, Columns: {n_cols}")

            # 5. Append summary info
            summary_rows.append({
                "task_id": task_id,
                "rows": n_rows,
                "columns": n_cols,
                "minority_class_pct": round(minority_pct, 2),
                "majority_class_pct": round(majority_pct, 2),
                "num_classes": num_classes,
                "target_name": target_name
            })

        except Exception as e:
            # If anything goes wrong for this task, print and continue
            print(f"Error processing task {task_id}: {e}")

    # 6. Sort summary rows by number of rows
    summary_rows.sort(key=lambda x: x["rows"])

    # 7. Save the summary CSV for all processed datasets
    summary_csv_path = os.path.join(output_dir, "tasks_summary.csv")
    with open(summary_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "rows", "columns", "minority_class_pct", "majority_class_pct", "num_classes", "target_name"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"\nSummary written to: {summary_csv_path}")
    print(f"Total datasets in CSV: {len(summary_rows)}")

if __name__ == "__main__":
    main()