import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

with open("model/random_forest_model.pkl", "rb") as model_file:
    model_random_forest = pickle.load(model_file)

FEATURES = [
    "HIDROGEN",
    "METHANE",
    "CARBON MONOXIDA",
    "CARBON DIOXIDA",
    "ETHYLENE",
    "ETHANE",
    "ACETHYLENE",
    "OKSIGEN",
    "NITROGEN",
    "TDCG",
]
LABEL = ["kondisi 0", "kondisi 1", "kondisi 2", "kondisi 3", "kondisi 4"]


@app.route("/")
def index():
    return {"status": "SUCCESS", "message": "Service is Up"}, 200


@app.route("/predict/pandas")
def predict_pandas():
    args = request.args
    hi = args.get("hi", default=0.0, type=float)
    me = args.get("me", default=0.0, type=float)
    cm = args.get("cm", default=0.0, type=float)
    cd = args.get("cd", default=0.0, type=float)
    et = args.get("et", default=0.0, type=float)
    eta = args.get("eta", default=0.0, type=float)
    ace = args.get("ace", default=0.0, type=float)
    ok = args.get("ok", default=0.0, type=float)
    ni = args.get("ni", default=0.0, type=float)
    tdcg = args.get("tdcg", default=0.0, type=float)
    new_data = [[hi, me, cm, cd, et, eta, ace, ok, ni, tdcg]]
    new_data = pd.DataFrame(new_data, columns=FEATURES)
    res = model_random_forest.predict(new_data)
    print("Nilai res:", res)

    res = LABEL[res[0]]

    return {
        "status": "SUCCESS",
        "input type": "Pandas DataFrame",
        "input": {
            "HIDROGEN": hi,
            "METHANE": me,
            "CARBON MONOXIDA": cm,
            "CARBON DIOXIDA": cd,
            "ETHYLENE": et,
            "ETHANE": eta,
            "ACETHYLENE": ace,
            "OKSIGEN": ok,
            "NITROGEN": ni,
            "TDCG": tdcg,
        },
        "result": res,
    }, 200


@app.route("/predict/post", methods=["POST"])
def predict_post():
    try:
        if request.form:
            # Jika data berupa form-data atau urlencoded
            data = {key: request.form[key] for key in FEATURES}
        else:
            # Jika data berupa JSON
            data = request.json

        new_data = pd.DataFrame([data], columns=FEATURES)
        res = model_random_forest.predict(new_data)
        print("Nilai res:", res)

        res_label = LABEL[res[0]]

        response_data = {
            "status": "SUCCESS",
            "input type": "Form Data" if request.form else "JSON Data",
            "input": data,
            "result": res_label,
        }

        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)

# app.run(debug=True)
