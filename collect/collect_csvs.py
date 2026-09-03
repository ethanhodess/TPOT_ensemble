import pandas as pd
import glob

# folder passed as argument
#path = "/common/hodesse/hpc_test/TPOT2_ensemble/logs/*.csv"
path = "/Users/ethanhodess/Documents/Documents - Ethan’s MacBook Pro/Cedars/2025/TPOT_ensemble/logs_random/*.csv"
files = glob.glob(path)

dfs = []

first_df = pd.read_csv(files[0])
dfs.append(first_df)

for f in files[1:]:
    df = pd.read_csv(f)
    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)
final_df.to_csv("bagging_res_1.csv", index=False)