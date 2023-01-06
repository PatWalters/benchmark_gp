#!/usr/bin/env python

import gpflow
from mol_gp import Tanimoto, TanimotoGP
from descriptastorus import descriptors
from glob import glob
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


def run_gp(input_df, descriptor_id, smiles_col="smiles", y_col="y"):
    parameter_dict = {
        "morgan": [["Morgan3"],
                   Tanimoto()],
        "morgan_rdkit": [["Morgan3", "RDKit2DNormalized"],
                         Tanimoto(active_dims=slice(1, 2049)) * gpflow.kernels.Matern52(active_dims=slice(2050, 2251))],
        "morgan_counts": [["Morgan3Counts"],
                          Tanimoto()],
        "morgan_counts_rdkit": [["Morgan3", "RDKit2DNormalized"],
                                Tanimoto(active_dims=slice(1, 2049)) * gpflow.kernels.Matern52(
                                    active_dims=slice(2050, 2251))]
    }
    parameter_list = parameter_dict.get(descriptor_id)
    if parameter_list is None:
        raise ValueError(f"{descriptor_id} is not a valid choice")
    feature_list, kernel = parameter_list
    # Generate features
    feature_generator = descriptors.MakeGenerator(feature_list)
    mols_processed, features = feature_generator.processSmiles(input_df[smiles_col])
    input_df['mol'] = mols_processed
    input_df['features'] = features
    # split training and test
    train = input_df.query("split == 'train'").copy()
    test = input_df.query("split == 'test'").copy()
    # fit and predict
    fitter = TanimotoGP(kernel)
    fitter.fit(np.stack(train.features), train[y_col])
    pred, uncertainty = fitter.predict(np.stack(test.features))
    test['pred'] = pred
    test['uncertainty'] = test[y_col]
    # return a dataframe with test results
    return test


def get_stats(df, y_col="y", pred_col="pred", name_col="dataset"):
    name = df[name_col].values[0]
    y = df[y_col]
    pred = df[pred_col]
    r2 = r2_score(y, pred)
    rmse = mean_squared_error(y, pred)
    print(name, r2, rmse)


def run_benchmark():
    for filename in glob("CHEMBL*.csv"):
        df = pd.read_csv(filename)
        df['dataset'] = filename.replace(".csv", "")
        res = run_gp(df, "morgan")
        get_stats(res)


if __name__ == "__main__":
    run_benchmark()
