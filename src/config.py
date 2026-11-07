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
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_ROOMS_DIR.mkdir(parents=True, exist_ok=True)

# Store Operating Parameters
STORE_NAME = "Apex Colors & Paints Studio"
STORE_SHORT_NAME = "Apex Colors & Paints"
STORE_TIMEZONE = os.getenv("STORE_TIMEZONE", "Asia/Kolkata")
STORE_OPEN_HOUR = 9     # 09:00 AM
STORE_CLOSE_HOUR = 21   # 09:00 PM
CURRENCY_SYMBOL = "₹"

# Demo mode: the seeded sample data is moved forward so its last day is always
# "today". Turn off (PAINT_SHOP_DEMO_MODE=0) once the shop runs on real data.
DEMO_MODE = os.getenv("PAINT_SHOP_DEMO_MODE", "1") not in ("0", "false", "False")

# Smart Alert Thresholds
EXPIRY_WARNING_DAYS = 45            # Alert for batches expiring within 45 days
DEAD_STOCK_DAYS = 60                # Look-back window for "money stuck" check
DEAD_STOCK_MAX_SALES = 2            # <= this many cans sold in the window = slow seller
DEAD_STOCK_MIN_UNITS = 10           # ...and at least this many cans on the shelf
CONVERSION_DROP_RATIO = 0.7         # Flag a day whose bills-per-visitor is < 70% of normal
CONVERSION_LOOKBACK_DAYS = 14

# Camera & Vision Configuration
DEFAULT_CAMERA_INDEX = 0            # Laptop/USB webcam
ESP32_DEFAULT_STREAM_URL = os.getenv("ESP32_STREAM_URL", "http://192.168.1.100:81/stream")
PERSON_CONFIDENCE_THRESHOLD = 0.5
RESTRICTED_ZONE_BOX = [0.68, 0.10, 0.96, 0.58]  # [x_min, y_min, x_max, y_max] normalised (storage room)
DOOR_LINE_Y = 0.68                  # Normalised height of the virtual door line
SECURITY_ALERT_COOLDOWN_SECONDS = 60  # At most one saved alert per type per minute

# Colour Studio — wall segmentation model (ONNX, runs on CPU)
WALL_MODEL_REPO = os.getenv("WALL_MODEL_REPO", "Xenova/segformer-b3-finetuned-ade-512-512")
WALL_MODEL_FILES = ["onnx/model_quantized.onnx", "onnx/model.onnx"]   # tried in this order
WALL_MODEL_PATH = os.getenv("WALL_MODEL_PATH", "")  # set to use a local .onnx instead of downloading
WALL_MODEL_INPUT_SIZE = 512
COLOUR_WORKING_MAX_SIDE = 960       # photos are resized to this for speed

# Paint quantity estimate. Replace with each product's sheet value when available.
DEFAULT_COVERAGE_SQFT_PER_LITRE_PER_COAT = {
    "Interior Emulsion": 110,
    "Exterior Emulsion": 90,
    "Enamel": 120,
    "Primer": 100,
    "Waterproofing": 45,
}

# Gemini Model Settings
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
