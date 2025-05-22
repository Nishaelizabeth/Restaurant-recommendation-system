from flask import Flask, render_template, request, jsonify
from routes import configure_routes

def create_app():
    app = Flask(__name__)
    configure_routes(app)
    return app

app = create_app()  # <-- ADD THIS

if __name__ == '__main__':
    app.run(debug=True)
