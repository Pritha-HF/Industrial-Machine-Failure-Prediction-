import json
import math
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from features import PhysicsFeatureEngineer

app = Flask(__name__)

model = joblib.load('industrial_failure_best_pipeline.joblib')
with open('model_metrics.json', 'r') as f:
    model_metrics = json.load(f)

@app.route('/')
def home():
    return render_template('index.html', metrics=model_metrics)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        m_type = str(data['type']).strip().upper()
        air_temp = float(data['air_temp'])
        proc_temp = float(data['proc_temp'])
        rot_speed = float(data['rot_speed'])
        torque = float(data['torque'])
        tool_wear = float(data['tool_wear'])

        input_df = pd.DataFrame([{
            'Type': m_type,
            'Air temperature [K]': air_temp,
            'Process temperature [K]': proc_temp,
            'Rotational speed [rpm]': rot_speed,
            'Torque [Nm]': torque,
            'Tool wear [min]': tool_wear
        }])

        pred = int(model.predict(input_df)[0])
        prob = float(model.predict_proba(input_df)[0][1]) * 100

        temp_diff = round(proc_temp - air_temp, 2)
        power_w = round(torque * (rot_speed * 2 * math.pi / 60), 1)
        overstrain = round(tool_wear * torque, 1)

        if prob >= 50:
            risk = "HIGH RISK"
        elif prob >= 20:
            risk = "MODERATE RISK"
        else:
            risk = "LOW RISK"

        return jsonify({
            'failure': pred == 1,
            'label': 'Machine Failure: Yes' if pred == 1 else 'Machine Failure: No',
            'probability': round(prob, 2),
            'risk': risk,
            'physics': {
                'temp_diff': temp_diff,
                'power_w': power_w,
                'overstrain': overstrain
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
