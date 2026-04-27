from flask import Flask, request, jsonify, render_template
from src.pipelines.prediction_pipeline import PredictionPipeline, CustomClass

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "IncomeIQ ML API",
        "version": "1.0.0",
        "model": "Income Predictor (Random Forest / Decision Tree / Logistic Regression)"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        json_data = request.get_json()

        data = CustomClass(
            age=int(json_data.get("age")),
            workclass=int(json_data.get("workclass")),
            education_num=int(json_data.get("education_num")),
            marital_status=int(json_data.get("marital_status")),
            occupation=int(json_data.get("occupation")),
            relationship=int(json_data.get("relationship")),
            race=int(json_data.get("race")),
            sex=int(json_data.get("sex")),
            capital_gain=int(json_data.get("capital_gain")),
            capital_loss=int(json_data.get("capital_loss")),
            hours_per_week=int(json_data.get("hours_per_week")),
            native_country=int(json_data.get("native_country"))
        )

        final_data = data.get_data_DataFrame()
        pipeline = PredictionPipeline()
        pred, confidence = pipeline.predict(final_data)

        return jsonify({
            "status": "success",
            "prediction": int(pred[0]),
            "income_category": "<=50K" if pred[0] == 0 else ">50K",
            "confidence": round(confidence, 2) if confidence is not None else None
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
