import cv2
import pytesseract
import numpy as np
import pandas as pd
from datetime import datetime
from mss import mss
from collections import deque
from sklearn.ensemble import IsolationForest
import os
import time

# === CONFIGURATION ===
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

ocr_region = {
    "top": 200,   # Y
    "left": 800,  # X
    "width": 200,
    "height": 70
}

CSV_LOG = "bustabit_live.csv"
MAX_HISTORY = 100
history = deque(maxlen=MAX_HISTORY)

def capture_multiplier(region):
    with mss() as sct:
        img = np.array(sct.grab(region))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        data = pytesseract.image_to_string(thresh, config='--psm 7')
        try:
            num = float(data.replace('x', '').replace('X', '').strip())
            return round(num, 2)
        except:
            return None

def append_to_csv(multiplier):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {'timestamp': now, 'multiplier': multiplier}
    df = pd.DataFrame([row])
    file_exists = os.path.isfile(CSV_LOG)
    df.to_csv(CSV_LOG, mode='a', header=not file_exists, index=False)

def detect_zone(multipliers):
    avg = np.mean(multipliers)
    low_count = np.sum(np.array(multipliers) < 2)
    high_count = np.sum(np.array(multipliers) >= 10)

    if low_count >= len(multipliers) * 0.8:
        return "🧊 ICE ZONE", "High crash density"
    elif avg > 4 and high_count >= 2:
        return "💎 BULLISH ZONE", "High multipliers showing up"
    elif avg < 2 and high_count == 0:
        return "🌊 REBOUND ZONE", "Possible big one coming"
    else:
        return "🔄 NEUTRAL ZONE", "No strong trend"

def predict_10x(multipliers):
    model = IsolationForest(contamination=0.01)
    X = np.array(multipliers).reshape(-1, 1)
    model.fit(X)
    prediction = model.predict([[1]])
    return "⚠️ POSSIBLE 10x+" if prediction[-1] == -1 else "Likely <10x"

def live_loop():
    print("🔁 Starting OCR capture loop...")
    last_seen = None
    while True:
        try:
            multiplier = capture_multiplier(ocr_region)
            if multiplier and multiplier != last_seen:
                last_seen = multiplier
                history.append(multiplier)
                append_to_csv(multiplier)

                if len(history) >= 10:
                    recent = list(history)[-30:]
                    zone, reason = detect_zone(recent)
                    prediction = predict_10x(recent)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {multiplier:.2f}x | Zone: {zone} | {prediction}")
        except Exception as e:
            print(f"[Error] {e}")
        time.sleep(1.5)

if __name__ == "__main__":
    live_loop()

