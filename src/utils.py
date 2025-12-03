from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import polars as pl


def custom_target_encode(data: pd.DataFrame, target: pd.Series, column: str) -> pl.Series:
    encoder = TargetEncoder()
    encoded = encoder.fit_transform(data[column], target)
    return encoded, encoder

def target_encode_nominal_columns(data: pl.DataFrame, target: pl.Series, columns: list) -> tuple[pl.DataFrame, dict[str, TargetEncoder]]:
    data_pandas = data.to_pandas()
    target_pandas = target.to_pandas()
    encoders = {}
    for col in columns:
        encoded, encoder = custom_target_encode(data_pandas, target_pandas, col)
        data_pandas[col] = encoded
        encoders[col] = encoder
    return pl.from_pandas(data_pandas), encoders

def custom_onehot_encode(data: pd.DataFrame, column: str) -> pl.DataFrame:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(data[[column]])
    encoded_df = pd.DataFrame(encoded, columns=[f"{column}_{cat}" for cat in encoder.categories_[0]])
    return encoded_df, encoder

def onehot_encode_nominal_columns(data: pl.DataFrame, columns: list) -> tuple[pl.DataFrame, dict[str, OneHotEncoder]]:
    data_pandas = data.to_pandas()
    encoders = {}
    encoded_dfs = []
    for col in columns:
        encoded_df, encoder = custom_onehot_encode(data_pandas, col)
        encoded_dfs.append(encoded_df)
        encoders[col] = encoder
    all_encoded_df = pd.concat(encoded_dfs, axis=1)
    data_pandas = pd.concat([data_pandas.reset_index(drop=True), all_encoded_df.reset_index(drop=True)], axis=1)
    data_pandas = data_pandas.drop(columns=columns)
    return pl.from_pandas(data_pandas), encoders