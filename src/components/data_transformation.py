import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


@dataclass
class DataTransformationConfig:
    # The fitted preprocessor is saved here so predictions can reuse the same scaling
    preprocess_obj_file_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info("Building preprocessing pipeline")

            numerical_features = [
                'age', 'workclass', 'education_num', 'marital_status',
                'occupation', 'relationship', 'race', 'sex', 'capital_gain',
                'capital_loss', 'hours_per_week', 'native_country'
            ]

            # Step 1: fill missing values with median, Step 2: scale to mean=0, std=1
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ])

            # Apply the pipeline to numerical columns only
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def remove_outliers_IQR(self, col, df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            upper_limit = Q3 + 1.5 * IQR
            lower_limit = Q1 - 1.5 * IQR

            # Cap values at the limits instead of removing them
            df.loc[df[col] > upper_limit, col] = upper_limit
            df.loc[df[col] < lower_limit, col] = lower_limit

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            preprocessing_obj = self.get_data_transformation_obj()

            target_column_name = "income"
            numerical_columns = [
                'age', 'workclass', 'education_num', 'marital_status',
                'occupation', 'relationship', 'race', 'sex', 'capital_gain',
                'capital_loss', 'hours_per_week', 'native_country'
            ]

            # Separate features (X) from label (y)
            input_feature_train_df = train_data.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_data[target_column_name]

            input_feature_test_df = test_data.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_data[target_column_name]

            # fit_transform on train (learns the scaling), transform on test (applies same scaling)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info("Preprocessing applied to train and test data")

            # Stack features and labels back together into one array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the fitted preprocessor so the prediction pipeline can reuse it
            save_object(
                file_path=self.data_transformation_config.preprocess_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessor object saved")

            return train_arr, test_arr, self.data_transformation_config.preprocess_obj_file_path

        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)
