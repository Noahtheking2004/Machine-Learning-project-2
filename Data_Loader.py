import os
import numpy as np
import pandas as pd


# Turn a DataFrame into X, y, is_cat (True = categorical column), names
def _prepare(df, target, categorical):
    y = df[target].to_numpy(float)
    X_cols, is_cat, names = [], [], []
    for col in df.columns:
        if col == target:
            continue
        if col in categorical:
            codes, _ = pd.factorize(df[col])      # text -> 0, 1, 2, ...
            X_cols.append(codes.astype(float))
            is_cat.append(True)
        else:
            X_cols.append(df[col].to_numpy(float))
            is_cat.append(False)
        names.append(col)
    return np.column_stack(X_cols), y, is_cat, names


# Load abalone, hardware, or forestfires from the data folder
def load_dataset(name, data_dir="data"):
    if name == "abalone":
        cols = ["sex", "length", "diameter", "height", "whole", "shucked",
                "viscera", "shell", "rings"]
        df = pd.read_csv(os.path.join(data_dir, "abalone.data"),
                         header=None, names=cols)
        return _prepare(df, "rings", {"sex"})

    if name == "hardware":
        cols = ["vendor", "model", "MYCT", "MMIN", "MMAX", "CACH",
                "CHMIN", "CHMAX", "PRP", "ERP"]
        df = pd.read_csv(os.path.join(data_dir, "machine.data"),
                         header=None, names=cols)
        # model = unique ID, ERP = the authors' own estimate of PRP
        df = df.drop(columns=["model", "ERP"])
        return _prepare(df, "PRP", {"vendor"})

    if name == "forestfires":
        df = pd.read_csv(os.path.join(data_dir, "forestfires.csv"))
        return _prepare(df, "area", {"month", "day"})

    raise ValueError("dataset must be abalone, hardware, or forestfires")