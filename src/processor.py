import joblib
import polars as pl
from .utils import prepare_input_for_prediction, scale_numerical_columns, scale_target, target_encode_nominal_columns, onehot_encode_nominal_columns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class Processor():
    def __init__(self, dataset_path, columns_to_drop, categorical_columns, nominal_columns, remaining_nominal_columns, numerical_columns):
        self.dataset_path = dataset_path
        self.columns_to_drop = columns_to_drop
        self.categorical_columns = categorical_columns
        self.nominal_columns = nominal_columns
        self.remaining_nominal_columns = remaining_nominal_columns
        self.numerical_columns = numerical_columns
        self.df = self.read_dataset()
        
        self.min_max_scalers = None
        self.target_scaler = None
        self.unique_values_dict = None
        self.target_encoders = None
        self.onehot_encoders = None

        self.model = None

    def read_dataset(self) -> pl.DataFrame | None:
        try:
            df = pl.read_csv(self.dataset_path)
            return df
        except Exception as e:
            return None

    def fix_null_values(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        df = self.df.drop(self.columns_to_drop)
        # drop nan categorical columns
        df = df.drop_nulls(subset=self.categorical_columns + ["price"])
        target = df["price"]
        data = df.drop("price")
        # add engineered features
        data = data.with_columns([
            (data["miles"] / (2023 - data["year"])).cast(pl.Int64).alias("miles_per_year"),
            (2023 - data["year"]).alias("age")
        ])
        # handle null values in numerical columns by filling with median
        data = data.with_columns([
            pl.col(col).fill_null(data[col].median()).alias(col)
            for col in self.numerical_columns
        ])
        
        return (data, target)
    
    def scale_numerical_columns(self, data: pl.DataFrame, target: pl.DataFrame) -> pl.DataFrame:
        data, min_max_scalers = scale_numerical_columns(data, self.numerical_columns)
        target, target_scaler = scale_target(target)

        self.min_max_scalers = min_max_scalers
        self.target_scaler = target_scaler

        target = pl.from_numpy(target.flatten()) 

        return data, target
    
    def find_unique_values(self, data: pl.DataFrame) -> None:
        unique_values_dict = {}
        for col in self.categorical_columns:
            unique_values = data[col].unique()
            unique_values_dict[col] = unique_values
        self.unique_values_dict = unique_values_dict


    def encode_categorical_features(self, data: pl.DataFrame, target: pl.DataFrame) -> pl.DataFrame:
        # convert nominal columns to target encoded columns
        target_encoded_data, target_encoders = target_encode_nominal_columns(data, target, self.nominal_columns)
        self.target_encoders = target_encoders
        # one-hot encode "transmission" and "drivetrain"
        onehot_encoded_data, onehot_encoders = onehot_encode_nominal_columns(target_encoded_data, self.remaining_nominal_columns)
        self.onehot_encoders = onehot_encoders

        return onehot_encoded_data
    
    def split_data(self, data: pl.DataFrame | pd.DataFrame, target: pl.DataFrame | pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        from sklearn.model_selection import train_test_split
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
            target = target.to_pandas()
        else:
            pass
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=random_state)
        return pl.from_pandas(X_train), pl.from_pandas(X_test), pl.from_pandas(y_train), pl.from_pandas(y_test)

    def train(self, X_train: pl.DataFrame, y_train: pl.DataFrame) -> LinearRegression:
        model = LinearRegression()
        self.model = model
        model.fit(X_train, y_train)
        return model
    
    def score(self, X_test: pl.DataFrame, y_test: pl.DataFrame) -> None:
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")

    def prepare_data_for_inference(self, inp: dict | pl.DataFrame, original_df:pl.DataFrame) -> pd.DataFrame:
        prepared_input = prepare_input_for_prediction(inp, numerical_columns=self.numerical_columns,
                                target_encoders=self.target_encoders, onehot_encoders=self.onehot_encoders, 
                                min_max_scalers=self.min_max_scalers, nominal_columns=self.nominal_columns, 
                                remaining_nominal_columns=self.remaining_nominal_columns,
                                numerical_medians={col: original_df[col].median() for col in self.numerical_columns})
        # Align prepared input columns and order with training features
        # Reindex will add any missing training columns as zeros and drop extras
        prepared_input = prepared_input.reindex(columns=original_df.columns, fill_value=0)
        # Convert to numeric (float) to match training dtypes used by the model
        prepared_input = prepared_input.astype(float)
        return prepared_input
    
    def preprocess(self):
        data, target = self.fix_null_values()
        data, target = self.scale_numerical_columns(data, target)
        self.find_unique_values(data)
        data = self.encode_categorical_features(data, target)
        X_train, X_test, y_train, y_test = self.split_data(data, target)
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate(self, X_train: pl.DataFrame, y_train: pl.DataFrame, X_test: pl.DataFrame, y_test: pl.DataFrame) -> None:
        model = self.train(X_train, y_train)
        self.score(X_test, y_test)
        # save model for inference
        self.model = model
        # Note: Model saving is now handled in the CLI (main.py)
    
    def infer(self, df: pd.DataFrame) -> float:
        predicted_price_scaled = self.model.predict(df)
        predicted_price = self.target_scaler.inverse_transform(predicted_price_scaled).item()
        return int(predicted_price)