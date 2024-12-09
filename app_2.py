from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
import json
app = Flask(__name__)



with open('GA_output_ET.json', 'r') as fp:
    feature_list = json.load(fp)



@app.route('/')
def index():
    return render_template('index_ws.html')  # Giao diện upload file

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
    features.remove('Label')


    # Trích xuất đặc trưng và thêm cột Timestamp
    df = data[features]  # `features` là danh sách cột cần dự đoán
    timestamps = data["Timestamp"]  # Giả sử file CSV có cột "Timestamp"

    # Dự đoán
    predictions = model.predict(df.values).tolist()

    # Tạo kết quả
    results = {
        "columns": features,  # Tên các cột đặc trưng
        "rows": [
            {"row": row.tolist(), "timestamp": timestamps.iloc[i], "prediction": predictions[i]}
            for i, row in enumerate(df.values)
        ]
    }

    return jsonify(results)  # Trả kết quả dưới dạng JSON

if __name__ == '__main__':
    app.run(debug=True)
