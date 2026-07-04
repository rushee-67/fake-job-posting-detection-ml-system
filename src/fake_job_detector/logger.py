import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create log file with timestamp
LOG_FILE = f"fake_job_detector_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_PATH,
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

# Logger object
logger = logging.getLogger("fake_job_detector")