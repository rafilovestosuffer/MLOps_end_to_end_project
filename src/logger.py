import os
import logging
from datetime import datetime

# Creates a unique filename like "04_25_2026_18_30_00.log"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Log files go into: logs/04_25_2026_18_30_00/
log_dir = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(log_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
