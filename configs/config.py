import os

PORT = 7860
MAX_POOL = 4
FLOAT16 = False

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/input'))
TEMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/temp'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/output'))
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
