from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow import keras

app = Flask(__name__)

# 모델과 전처리 파이프라인 로드
model = keras.models.load_model("forest_fire_model.keras")
pipeline = joblib.load("forest_fire_pipeline.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':
        # 폼에서 사용자 입력 가져오기
        longitude = float(request.form['longitude'])
        latitude = float(request.form['latitude'])
        month = request.form['month']
        day = request.form['day']
        avg_temp = float(request.form['avg_temp'])
        max_temp = float(request.form['max_temp'])
        max_wind_speed = float(request.form['max_wind_speed'])
        avg_wind = float(request.form['avg_wind'])
        
        # 입력 데이터 전처리
        input_data = {
            'longitude': longitude,
            'latitude': latitude,
            'month': month,
            'day': day,
            'avg_temp': avg_temp,
            'max_temp': max_temp,
            'max_wind_speed': max_wind_speed,
            'avg_wind': avg_wind
        }
        
        # 딕셔너리를 데이터프레임으로 변환
        import pandas as pd
        input_df = pd.DataFrame([input_data])
        
        # 파이프라인을 사용하여 전처리
        input_prepared = pipeline.transform(input_df)
        
        # 예측
        prediction = model.predict(input_prepared)[0][0]
        
        # 로그 변환된 값을 원래 스케일로 변환
        original_scale_prediction = np.exp(prediction) - 1
        
        # 2) 헥타르 → 제곱미터
        pred_m2 =  original_scale_prediction * 10000
        
        return render_template('result.html', 
                               prediction=pred_m2,
                               input_data=input_data)

if __name__ == '__main__':
    app.run(debug=True)