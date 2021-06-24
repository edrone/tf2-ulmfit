import argparse

import pandas as pd
from sklearn.metrics import mean_absolute_error


def ar_score(y_true, y_pred):
    ds = pd.DataFrame({
        'y_true': (y_true - 1.0)/4.0,
        'y_pred': (y_pred - 1.0)/4.0,
    })
    wmae = ds \
        .groupby('y_true') \
        .apply(lambda df: mean_absolute_error(df['y_true'], df['y_pred']))
    print(wmae)
    wmae = wmae.mean()
    print(f"AR = {1 - wmae}")
    return 1 - wmae

if __name__ == "__main__":
    argz = argparse.ArgumentParser()
    argz.add_argument("--results-tsv", required=True, help="Regressor results file")
    argz = vars(argz.parse_args())
    results = pd.read_csv(argz['results_tsv'], sep='\t')
    ar_score(results['gold'], results['y_preds'])
