from flask import Flask, jsonify, request

import ml

app = Flask(__name__)

_artifacts = None


def get_artifacts():
    global _artifacts
    if _artifacts is None:
        try:
            _artifacts = ml.load_artifacts()
        except FileNotFoundError:
            return None
    return _artifacts


@app.post("/api/sentiment")
def analyze_sentiment():
    payload = request.get_json(silent=True)

    if not isinstance(payload, list):
        return jsonify({
            "error": "Le corps de la requête doit être un tableau de chaînes (string[])."
        }), 400

    if len(payload) == 0:
        return jsonify({"error": "La liste de tweets ne peut pas être vide."}), 400

    if not all(isinstance(tweet, str) for tweet in payload):
        return jsonify({
            "error": "Chaque élément de la liste doit être une chaîne de caractères."
        }), 400

    artifacts = get_artifacts()
    if artifacts is None:
        return jsonify({
            "error": "Le modèle n'a pas encore été entraîné. Lancez `python train_model.py`."
        }), 503

    vectorizer, positive_model, negative_model = artifacts
    scores = ml.score_texts(vectorizer, positive_model, negative_model, payload)

    result = {tweet: round(score, 4) for tweet, score in zip(payload, scores)}
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(debug=True)
