"""
Database Models for Student Performance Predictor
Defines SQLAlchemy ORM models for storing student data, predictions, and improvement plans.
"""

import json
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
# Attempt to import JSONB for PostgreSQL optimization
try:
    from sqlalchemy.dialects.postgresql import JSONB
    JSON_COLUMN_TYPE = JSONB
except ImportError:
    # Fallback to standard SQLAlchemy JSON or db.JSON (which usually resolves to JSON on SQLite)
    JSON_COLUMN_TYPE = SQLAlchemy().JSON 
    
# Initialize SQLAlchemy database instance
db = SQLAlchemy()

class Student(db.Model):
    """
    Student model to store student information, features, prediction results, 
    and audit metadata in a single table.
    """
    
    __tablename__ = 'students'
    
    # Primary key
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    
    # Student Information (from file)
    name = db.Column(db.String(255), nullable=False, index=True)
    # Using 'year' and 'semester' as per your original model
    year = db.Column(db.Integer, nullable=False) 
    semester = db.Column(db.Integer, nullable=False)
    
    # Student features as JSONB/JSON
    features = db.Column(JSON_COLUMN_TYPE, nullable=False) 
    
    # Prediction results
    predicted_performance = db.Column(db.String(50), nullable=True) # e.g., 'Success', 'Failure'
    improvement_plan = db.Column(db.Text, nullable=True)
    
    # Audit/Logging Columns
    prediction_model_version = db.Column(db.String(50), nullable=True)
    confidence_score = db.Column(db.Float, nullable=True)
    
    # Metadata
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        """String representation of Student object"""
        return f'<Student {self.id}: {self.name} (Year {self.year})>'
    
    def to_dict(self):
        """
        Convert Student object to dictionary for JSON serialization.
        """
        # SQLAlchemy handles JSON/JSONB conversion automatically on retrieval. 
        # The complex serialization logic is often unnecessary unless using older versions/drivers.
        features_data = self.features if self.features is not None else {}
        
        # Ensure 'features' is a dict for serialization if it's stored as a string JSON
        if isinstance(features_data, str):
            try:
                features_data = json.loads(features_data)
            except json.JSONDecodeError:
                features_data = {}

        return {
            'id': self.id,
            'name': self.name,
            'year': self.year,
            'semester': self.semester,
            'features': features_data,
            'predicted_performance': self.predicted_performance,
            'improvement_plan': self.improvement_plan,
            'prediction_model_version': self.prediction_model_version,
            'confidence_score': self.confidence_score,
            # Use isoformat() for date serialization
            'created_at': self.created_at.isoformat(), 
            'updated_at': self.updated_at.isoformat()
        }

def init_db(app):
    """
    Initialize database tables within the Flask app context.
    Creates all tables defined in the models if they don't already exist.
    """
    with app.app_context():
        try:
            db.create_all()
            # Optionally check if tables exist to confirm successful creation
            # logger.info("Database tables checked/created successfully.")
        except Exception as e:
            # IMPORTANT: Log this error instead of just printing in a production environment
            print(f"FATAL ERROR initializing database tables: {str(e)}") 
            # Re-raise the exception or handle appropriately if table creation is critical