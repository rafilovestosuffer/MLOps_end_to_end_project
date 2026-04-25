from src.pipelines.training_pipeline import TrainingPipeline
from src.logger import logging

if __name__ == "__main__":
    logging.info("Starting the training pipeline")
    pipeline = TrainingPipeline()
    accuracy = pipeline.run()
    print(f"\nTraining complete. Best model accuracy: {accuracy:.4f}")
