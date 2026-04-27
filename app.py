import os
import sys
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

ARTIFACTS_EXIST = (
    os.path.exists("artifacts/model_trainer/model.pkl") and
    os.path.exists("artifacts/data_transformation/preprocessor.pkl")
)


def _demo_predict(p: dict):
    """Heuristic fallback used when model artifacts are not yet trained."""
    score = 0
    if p["age"] >= 40:                   score += 2
    if p["education_num"] >= 13:         score += 3   # Bachelors+
    if p["occupation"] in [3, 9, 12]:   score += 3   # Exec / Prof / Tech
    if p["capital_gain"] > 5000:         score += 4
    if p["hours_per_week"] >= 45:        score += 1
    if p["marital_status"] in [1, 2]:    score += 1   # Married
    pred = 1 if score >= 6 else 0
    confidence = min(50 + score * 5, 89)
    return pred, float(confidence)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "IncomeIQ ML API",
        "version": "1.0.0",
        "model_trained": ARTIFACTS_EXIST,
        "mode": "production" if ARTIFACTS_EXIST else "demo"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        json_data = request.get_json()

        payload = {
            "age":            int(json_data.get("age")),
            "workclass":      int(json_data.get("workclass")),
            "education_num":  int(json_data.get("education_num")),
            "marital_status": int(json_data.get("marital_status")),
            "occupation":     int(json_data.get("occupation")),
            "relationship":   int(json_data.get("relationship")),
            "race":           int(json_data.get("race")),
            "sex":            int(json_data.get("sex")),
            "capital_gain":   int(json_data.get("capital_gain")),
            "capital_loss":   int(json_data.get("capital_loss")),
            "hours_per_week": int(json_data.get("hours_per_week")),
            "native_country": int(json_data.get("native_country")),
        }

        demo_mode = False
        if ARTIFACTS_EXIST:
            from src.pipelines.prediction_pipeline import PredictionPipeline, CustomClass
            data = CustomClass(**payload)
            df = data.get_data_DataFrame()
            pipeline = PredictionPipeline()
            pred, confidence = pipeline.predict(df)
            pred = int(pred[0])
        else:
            pred, confidence = _demo_predict(payload)
            demo_mode = True

        return jsonify({
            "status":         "success",
            "prediction":     pred,
            "income_category": "<=50K" if pred == 0 else ">50K",
            "confidence":     round(confidence, 2) if confidence is not None else None,
            "demo_mode":      demo_mode
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
