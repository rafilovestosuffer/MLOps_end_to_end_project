import os

def create_files_and_directories(file_list):
    for filepath in file_list:
        filedir = os.path.dirname(filepath)
        filename = os.path.basename(filepath)

        if filedir and not os.path.isdir(filedir):
            os.makedirs(filedir)
            print(f"Creating directory: {filedir} for the file {filename}")

        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            open(filepath, 'a').close()
            print(f"Creating empty file: {filepath}")
        else:
            print(f"{filename} already exists")

list_of_files = [
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_monitoring.py",
    "src/pipelines/__init__.py",
    "src/pipelines/training_pipeline.py",
    "src/pipelines/prediction_pipeline.py",
    "src/exception.py",
    "src/logger.py",
    "src/utils.py",
    "main.py"
]

create_files_and_directories(list_of_files)
