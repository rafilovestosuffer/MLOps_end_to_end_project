from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging


class TrainingPipeline:
    def run(self):
        # Step 1: Load raw data and split into train/test CSVs
        logging.info("=== STAGE 1: Data Ingestion ===")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Step 2: Preprocess features and save the preprocessor
        logging.info("=== STAGE 2: Data Transformation ===")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )

        # Step 3: Train models, pick the best, save it
        logging.info("=== STAGE 3: Model Training ===")
        model_trainer = ModelTrainer()
        accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"Training pipeline complete. Best accuracy: {accuracy}")
        return accuracy
