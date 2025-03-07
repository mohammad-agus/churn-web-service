import pickle
from flask import Flask, request, jsonify


input_file = 'model_C=1.0.bin'
with open(input_file, 'rb') as f:
    dv, model = pickle.load(f)

app = Flask('churn-web-service')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        "churn": bool(churn),
        "churn_probability": float(y_pred)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
    





