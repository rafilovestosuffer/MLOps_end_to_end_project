from flask import Flask, request, jsonify
from src.pipelines.prediction_pipeline import PredictionPipeline, CustomClass

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        json_data = request.get_json()

        # Map incoming JSON fields to our CustomClass
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
        pred = pipeline.predict(final_data)

        return jsonify({
            "status": "success",
            "prediction": int(pred[0]),
            "income_category": "<=50K" if pred[0] == 0 else ">50K"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
