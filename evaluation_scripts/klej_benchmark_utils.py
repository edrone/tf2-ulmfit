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
    argz.add_argument("--preds-normalized", action='store_true', help="Set this is --normalize-labels was used for training.")
    argz = vars(argz.parse_args())
    df = pd.read_csv(argz['results_tsv'], sep='\t')
    # clip to min and max values:
    if argz.get('preds_normalized'):
        df.loc[df['y_preds'] < 0, 'y_preds'] = 0.0
        df.loc[df['y_preds'] > 1, 'y_preds'] = 1.0
    df.loc[df['y_preds_rescaled'] > 5, 'y_preds_rescaled'] = 5.0
    df.loc[df['y_preds_rescaled'] < 1, 'y_preds_rescaled'] = 1.0

    ar_score(df['gold_unscaled'], df['y_preds_rescaled'])
