import pickle
import xgboost as xgb
from flask import Flask
from flask import request  
from flask import jsonify

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


app = Flask('diagnosis')  

@app.route('/predict',methods = ['POST']) 
def predict():
    data = request.get_json()
    X = dv.transform([data])
    feature_names = list(dv.get_feature_names_out())
    dtest = xgb.DMatrix(X, feature_names=feature_names)
    y_pred = model.predict(dtest)
    diagnosis = y_pred >= 0.5

    result = {
        'diag_probability': float(y_pred),
        'diagnosis': bool(diagnosis)
    } 

    return jsonify(result)


if __name__ == "__main__":  
    app.run(debug = True, host = '0.0.0.0', port = 9696) 