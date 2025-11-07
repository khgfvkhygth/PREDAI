
from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

app = Flask(__name__)
CSV_PATH = 'bustabit_live.csv'

def load_data():
    try:
        df = pd.read_csv(CSV_PATH)
        df = df.tail(100)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except:
        return pd.DataFrame(columns=['timestamp', 'multiplier'])

def detect_zone(multipliers):
    avg = np.mean(multipliers)
    low_count = np.sum(np.array(multipliers) < 2)
    high_count = np.sum(np.array(multipliers) >= 10)

    if low_count >= len(multipliers) * 0.8:
        return "🧊 ICE ZONE"
    elif avg > 4 and high_count >= 2:
        return "💎 BULLISH ZONE"
    elif avg < 2 and high_count == 0:
        return "🌊 REBOUND ZONE"
    else:
        return "🔄 NEUTRAL ZONE"

def predict_10x(multipliers):
    model = IsolationForest(contamination=0.01)
    X = np.array(multipliers).reshape(-1, 1)
    model.fit(X)
    pred = model.predict([[1]])
    return "⚠️ POSSIBLE 10x+" if pred[-1] == -1 else "Likely <10x"

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/data')
def data():
    df = load_data()
    if df.empty:
        return jsonify({"labels": [], "values": [], "zone": "N/A", "prediction": "N/A"})

    multipliers = df['multiplier'].values.tolist()
    timestamps = df['timestamp'].dt.strftime('%H:%M:%S').tolist()
    zone = detect_zone(multipliers)
    prediction = predict_10x(multipliers)

    return jsonify({
        "labels": timestamps,
        "values": multipliers,
        "zone": zone,
        "prediction": prediction
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
