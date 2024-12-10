from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
import json
from enum import Enum


app = Flask(__name__)

class DDoSType(Enum):
    SYN = "SYN Flood"
    HTTP = "HTTP Flood"
    ACK = "ACK Flood"
    UDP = "UDP Flood"
    ARP = "ARP Spoofing"
    SP = "Port Scanning"
    BF = "Brute Force"

with open('GA_output_ET.json', 'r') as fp:
    feature_list = json.load(fp)



@app.route('/')
def index():
    return render_template('index_ws.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Nhận file CSV từ client
    file = request.files['file']
    data = pd.read_csv(file)
    attack_type = request.form.get('attack_type')

    model_path = f'./models/NB_{attack_type}_model.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    features = feature_list[attack_type]
    if 'Label' in features:
        features.remove('Label')


    df = data[features]
    timestamps = data["Timestamp"]

    # Dự đoán
    predictions = model.predict(df.values).tolist()

    # Tạo kết quả
    results = {
        "columns": features,
        "rows": [
            {
                "row": row.tolist(),
                "timestamp": timestamps.iloc[i],
                "prediction": DDoSType[attack_type].value if predictions[i] == 1 else "Benign"
            }
            for i, row in enumerate(df.values)
        ]
    }

    return jsonify(results)  # Trả kết quả dưới dạng JSON

if __name__ == '__main__':
    app.run(debug=True)
