"""
System Configuration & Business Rules for Paint Shop AI Operating System
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "paint_shop.db"
SHADES_CATALOG_PATH = DATA_DIR / "shades_catalog.json"
SAMPLE_ROOMS_DIR = DATA_DIR / "sample_rooms"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_ROOMS_DIR.mkdir(parents=True, exist_ok=True)

# Store Operating Parameters
STORE_NAME = "Apex Colors & Paints Studio"
STORE_OPEN_HOUR = 9     # 09:00 AM
STORE_CLOSE_HOUR = 21   # 09:00 PM
CURRENCY_SYMBOL = "₹"

# Smart Alert Thresholds
LOW_STOCK_DAYS_THRESHOLD = 7        # Stock expected to run out within 7 days
EXPIRY_WARNING_DAYS = 45            # Alert for batches expiring within 45 days
OVERSTOCK_TURNOVER_DAYS = 60        # Flag as overstock if holding > 60 days forward demand
DEAD_STOCK_DAYS = 60                # No sales recorded for 60 days
SURGE_VELOCITY_MULTIPLIER = 1.8     # Sales spike > 1.8x baseline velocity

# Camera & Vision Configuration
DEFAULT_CAMERA_INDEX = 0            # Laptop/USB webcam
ESP32_DEFAULT_STREAM_URL = "http://192.168.1.100:81/stream"
PERSON_CONFIDENCE_THRESHOLD = 0.5
RESTRICTED_ZONE_BOX = [0.65, 0.1, 0.95, 0.6]  # [x_min, y_min, x_max, y_max] normalized coordinates (Storage Room)

# Gemini Model Settings
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
