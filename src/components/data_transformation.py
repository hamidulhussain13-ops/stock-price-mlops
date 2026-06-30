import os
import sys

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        "artifact",
        "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        try:

            target_column_name = "SalePrice"

            numerical_columns = [
                "LotFrontage","LotArea","OverallQual","OverallCond",
                "YearBuilt","YearRemodAdd","MasVnrArea","BsmtFinSF1",
                "BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF",
                "2ndFlrSF","GrLivArea","FullBath","HalfBath",
                "BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd",
                "Fireplaces","GarageCars","GarageArea",
                "WoodDeckSF","OpenPorchSF","EnclosedPorch",
                "ScreenPorch","PoolArea","MiscVal",
                "MoSold","YrSold"
            ]

            categorical_columns = [
                "MSZoning","Street","LotShape","LandContour",
                "Utilities","LotConfig","Neighborhood",
                "HouseStyle","RoofStyle","Exterior1st",
                "Exterior2nd","Foundation","Heating",
                "CentralAir","KitchenQual","SaleType",
                "SaleCondition"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Creating preprocessing object")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, data):

        try:

            logging.info("Data Transformation Started")

            target_column_name = "SalePrice"

            X = data.drop(columns=[target_column_name], axis=1)
            y = data[target_column_name]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            preprocessing_obj = self.get_data_transformer_object()

            X_train = preprocessing_obj.fit_transform(X_train)
            X_test = preprocessing_obj.transform(X_test)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor Saved Successfully")

            return (
                X_train,
                X_test,
                y_train,
                y_test
            )

        except Exception as e:
            raise CustomException(e, sys)