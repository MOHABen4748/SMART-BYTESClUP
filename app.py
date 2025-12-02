from flask import Flask, render_template, request
from model_utils import predict_news

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    
    if request.method == "POST":
        text = request.form.get("news_text")
        result = predict_news(text)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
