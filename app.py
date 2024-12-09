from flask import Flask, render_template, request
import pandas as pd
import os
import numpy as np
import json
import pickle
app = Flask(__name__)

# Tạo thư mục tạm để lưu file tải lên
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():

    if 'file' not in request.files or 'attack_type' not in request.form:
        return "Missing file or attack type", 400

    file = request.files['file']
    attack_type = request.form.get('attack_type')

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    # Kiểm tra xem file có phải định dạng CSV không
    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)  # Lưu file tạm thời

        # Đọc file CSV vào DataFrame
        df = pd.read_csv(file_path)

        # Đường dẫn tới file model
        model_path = "models/NB_BF_0_model.pkl"

        # Load mô hình
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Đường dẫn tới file dữ liệu mới
        data_path = "./csv_flow/Telnet Brute Force_test.csv"

        # Đọc dữ liệu
        file_df = pd.read_csv(data_path)

        with open('GA_output_ET.json', 'r') as fp:
            feature_list = json.load(fp)

        # Xử lý dữ liệu: Chọn các cột đặc trưng đã dùng khi train
        features = feature_list["BF"]

        features.remove('Label')

        X_new = file_df[features]

        # Xử lý giá trị thiếu (nếu có)
        X_new = X_new.fillna(0)

        # Chuyển đổi về numpy array (nếu cần)
        X_new = np.array(X_new)
        # Dự đoán nhãn cho dữ liệu mới
        predictions = model.predict(X_new)

        # Dự đoán xác suất (nếu cần)
        # probabilities = model.predict_proba(X_new)

        # Hiển thị kết quả
        # print("Predictions:", predictions)
        # print("Probabilities:", probabilities)

        # Tạo DataFrame chứa kết quả
        output_df = file_df[features].copy()  # Sao chép dữ liệu gốc nếu cần lưu thông tin ban đầu
        output_df['Label'] = predictions
        # Xử lý DataFrame (ví dụ: hiển thị thông tin)
        return render_template('result.html',attack_type=attack_type, table_html=output_df.to_html(classes='table table-bordered table-striped'))
    else:
        return "Please upload a CSV file", 400

if __name__ == '__main__':
    app.run(debug=True)
