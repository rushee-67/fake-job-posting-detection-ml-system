from flask import Flask, render_template, request

from fake_job_detector.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

pipeline = PredictionPipeline()


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None

    if request.method == "POST":
        prediction = pipeline.predict(
            request.form["title"],
            request.form["company_profile"],
            request.form["description"],
            request.form["requirements"],
            request.form["benefits"]
        )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )