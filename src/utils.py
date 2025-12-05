from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import polars as pl


def custom_target_encode(data: pd.DataFrame, target: pd.Series, column: str) -> tuple[pl.DataFrame, TargetEncoder]:
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

def custom_onehot_encode(data: pd.DataFrame, column: str) -> tuple[pl.DataFrame, OneHotEncoder]:
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


def scale_column(data: pd.DataFrame, column: str) -> tuple[pl.Series, MinMaxScaler]:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[[column]])
    return scaled, scaler

def scale_numerical_columns(data: pl.DataFrame, columns: list) -> tuple[pl.DataFrame, dict[str, MinMaxScaler]]:
    scalers = {}
    data_pandas = data.to_pandas()
    for col in columns:
        scaled, scaler = scale_column(data_pandas, col)
        data_pandas[col] = scaled
        scalers[col] = scaler
    return pl.from_pandas(data_pandas), scalers

def scale_target(target: pl.Series) -> tuple[pl.Series, MinMaxScaler]:
    scaler = MinMaxScaler()
    scaled_target = scaler.fit_transform(target.to_frame())
    return scaled_target, scaler


def prepare_input_for_prediction(
    new_data,
    numerical_columns: list,
    nominal_columns: list,
    remaining_nominal_columns: list,
    min_max_scalers: dict,
    target_encoders: dict,
    onehot_encoders: dict,
    numerical_medians: dict | None = None,
):
    """
    Prepare new input data for prediction by applying the same preprocessing
    used during training.

    Parameters
    - new_data: dict, pandas.DataFrame, or polars.DataFrame containing one or
      more rows to prepare.
    - numerical_columns: list of numerical column names that should be scaled.
    - nominal_columns: list of nominal columns that were target-encoded.
    - remaining_nominal_columns: list of nominal columns that were one-hot encoded.
    - min_max_scalers: dict mapping numerical column -> fitted MinMaxScaler.
    - target_encoders: dict mapping nominal column -> fitted TargetEncoder.
    - onehot_encoders: dict mapping nominal column -> fitted OneHotEncoder.
    - numerical_medians: dict mapping numerical column -> median value from training.

    Returns
    - pandas.DataFrame with processed features ready to pass to a model's
      `predict` method (column set matches training-time processed DataFrame).

    Example
    -------
    >>> df_ready = prepare_input_for_prediction(
    ...     new_data={'year':2020,'miles':30000,...},
    ...     numerical_columns=numerical_columns,
    ...     nominal_columns=nominal_columns,
    ...     remaining_nominal_columns=remaining_nominal_columns,
    ...     min_max_scalers=min_max_scalers,
    ...     target_encoders=target_encoders,
    ...     onehot_encoders=onehot_encoders,
    ...     numerical_medians=numerical_medians,
    ... )
    """

    # convert input to pandas DataFrame
    if isinstance(new_data, pl.DataFrame):
        df = new_data.to_pandas()
    elif isinstance(new_data, dict):
        df = pd.DataFrame([new_data])
    elif isinstance(new_data, pd.DataFrame):
        df = new_data.copy()
    else:
        raise TypeError("new_data must be dict, pandas.DataFrame, or polars.DataFrame")

    # drop columns that were removed during training if present
    to_drop = ["id", "vin", "stock_no", "seller_name", "street", "zip"]
    for c in to_drop:
        if c in df.columns:
            df = df.drop(columns=[c])

    # ensure expected columns exist (create with NaN if missing)
    expected_cols = set(numerical_columns + nominal_columns + remaining_nominal_columns)
    for c in expected_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # feature engineering: miles_per_year and age
    # protect against division by zero (year == 2023)
    if "year" in df.columns and "miles" in df.columns:
        denom = 2023 - df["year"]
        # replace 0 with 1 to avoid division by zero; keeps miles_per_year == miles
        denom = denom.replace(0, 1)
        df["miles_per_year"] = (df["miles"] / denom).astype("Int64")
        df["age"] = (2023 - df["year"]).astype("Int64")
    else:
        # create columns if not present
        if "miles_per_year" not in df.columns:
            df["miles_per_year"] = pd.NA
        if "age" not in df.columns:
            df["age"] = pd.NA

    # fill numeric nulls using provided medians
    if numerical_medians is None:
        raise ValueError("numerical_medians dict is required to fill numeric nulls. Pass training medians.")

    for col in numerical_columns:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].fillna(numerical_medians.get(col))

    # also ensure engineered features filled
    for col in ["miles_per_year", "age"]:
        if col in df.columns:
            df[col] = df[col].fillna(numerical_medians.get(col, 0))

    # apply target encoders to nominal columns
    for col in nominal_columns:
        encoder = target_encoders.get(col)
        if encoder is None:
            raise KeyError(f"No target encoder found for column '{col}'")
        # encoder.transform may accept DataFrame or Series; handle both
        try:
            encoded = encoder.transform(df[[col]])
        except Exception:
            encoded = encoder.transform(df[col])
        if isinstance(encoded, pd.DataFrame):
            df[col] = encoded.iloc[:, 0]
        else:
            df[col] = encoded

    # apply one-hot encoders for remaining nominal columns
    for col in remaining_nominal_columns:
        encoder = onehot_encoders.get(col)
        if encoder is None:
            raise KeyError(f"No one-hot encoder found for column '{col}'")
        arr = encoder.transform(df[[col]])
        # column names from fitted encoder categories_
        cats = encoder.categories_[0]
        ohe_cols = [f"{col}_{c}" for c in cats]
        ohe_df = pd.DataFrame(arr, columns=ohe_cols, index=df.index)
        df = pd.concat([df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
        df = df.drop(columns=[col])

    # scale numerical columns using provided scalers
    for col in numerical_columns:
        scaler = min_max_scalers.get(col)
        if scaler is None:
            raise KeyError(f"No scaler found for numerical column '{col}'")
        scaled = scaler.transform(df[[col]])
        # flattened assignment
        df[col] = [v[0] for v in scaled]

    # ensure deterministic column order: sort columns (user may adjust later)
    df = df.reindex(sorted(df.columns), axis=1)

    return df