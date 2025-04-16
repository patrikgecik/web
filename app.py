from flask import Flask, request, jsonify
from predict_bert import predict
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "en")
    
    result = predict(text, lang)
    return jsonify({"result": result})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
