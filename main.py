from flask import Flask, request, jsonify
from flask_cors import CORS
from socket_manager import socketio
from agent import run_chatbot

app = Flask(__name__)
CORS(app)


socketio.init_app(app)

@app.route("/search", methods=["POST"])
def search():

    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({
            "success": False,
            "message": "Query is required"
        }), 400

    try:
        answer = run_chatbot(data["query"])

        return jsonify({
            "success": True,
            "answer": answer
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


if __name__ == "__main__":
   socketio.run(app, host="0.0.0.0", port=5000, debug=True)