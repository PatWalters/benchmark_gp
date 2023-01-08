#!/usr/bin/env python

import gpflow
from mol_gp import Tanimoto, TanimotoGP
from descriptastorus import descriptors
from glob import glob
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import os
# Tell TensorFlow to shut the hell up
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# This isn't the most efficient way to implement things. I realize I'm caluclating the same descriptors multiple times.
# In my defense, I wanted to make sure I did this right so I tried to keep things as simple as possible. 


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
    return [name, r2, rmse]


def run_benchmark():
    row_list = []
    for filename in glob("data/CHEMBL*.csv"):
        df = pd.read_csv(filename)
        df['dataset'] = filename.replace(".csv", "").replace("data/","")
        for desc in ["morgan","morgan_rdkit","morgan_counts","morgan_counts_rdkit"]:
            res = run_gp(df, desc)
            name, r2, rmse = get_stats(res)
            _,r2_cliff,rmse_cliff = get_stats(res.query("cliff_mol == 1"))            print(name,desc,r2,rmse,r2_cliff,rmse_cliff)
            row_list.append([name,desc,r2,rmse,r2_cliff,rmse_cliff])
    row_df = pd.DataFrame(row_list,columns=["Name","Descriptors","R2","RMSE","R2_cliff","RMSE_cliff"])
    row_df.to_csv("molecule_ace_gp_results.csv",index=False)

if __name__ == "__main__":
    run_benchmark()
