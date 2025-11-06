"""
Student Performance Predictor - Flask Application
Main entry point for the backend API server.
Handles single and batch student performance predictions with personalized improvement plans.
"""

import os
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # Enable CORS for frontend requests

# Configuration
app.config["JSON_SORT_KEYS"] = False
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL",
    "sqlite:///student_predictions.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize database
from models import db, init_db
db.init_app(app)

# Import API routes
from routes import api_bp

# ✅ Frontend route
@app.route("/")
def index():
    return render_template("index_updated.html")

# ✅ Serve static files (CSS, JS, etc.)
@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

# Register API blueprint
app.register_blueprint(api_bp, url_prefix="/api")

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Student Performance Predictor API"
    }), 200

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    logger.warning(f"Bad request: {str(error)}")
    return jsonify({
        "error": "Bad Request",
        "message": str(error),
        "timestamp": datetime.utcnow().isoformat()
    }), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not Found",
        "message": "The requested resource was not found",
        "timestamp": datetime.utcnow().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.utcnow().isoformat()
    }), 500

# Ensure database is initialized before handling requests
@app.before_request
def before_request():
    with app.app_context():
        init_db()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=os.getenv("FLASK_ENV") == "development"
    )
