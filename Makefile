install:
	pip install -r requirements.txt
	pip install -e .

train:
	python main.py

serve:
	python app.py

mlflow:
	mlflow server --host 0.0.0.0 --port 5000

docker-build:
	docker build -t mlops-project .

docker-run:
	docker-compose up

test:
	pytest tests/ -v

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache

.PHONY: install train serve mlflow docker-build docker-run test clean
