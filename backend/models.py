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
    
    # Student Information (from file/user input)
    name = db.Column(db.String(255), nullable=False, index=True) # User entered name
    year = db.Column(db.Integer, nullable=False) 
    major = db.Column(db.String(100), nullable=False) # ✨ NEW FIELD: major
    
    # Student features as JSONB/JSON
    features = db.Column(JSON_COLUMN_TYPE, nullable=False) 
    
    # Prediction results
    predicted_performance = db.Column(db.String(50), nullable=True) # e.g., 'Success', 'Failure'
    improvement_plan = db.Column(db.Text, nullable=True)
    
    # Metadata
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # ❌ REMOVED: semester, updated_at, prediction_model_version, confidence_score
    
    def __repr__(self):
        """String representation of Student object"""
        return f'<Student {self.id}: {self.name} (Year {self.year})>'
    
    def to_dict(self):
        """
        Convert Student object to dictionary for JSON serialization.
        """
        features_data = self.features if self.features is not None else {}
        
        if isinstance(features_data, str):
            try:
                features_data = json.loads(features_data)
            except json.JSONDecodeError:
                features_data = {}

        return {
            'id': self.id,
            'name': self.name,
            'year': self.year,
            'major': self.major, # ✨ NEW FIELD: major
            'features': features_data,
            'predicted_performance': self.predicted_performance,
            'improvement_plan': self.improvement_plan,
            'created_at': self.created_at.isoformat()
            # ❌ REMOVED fields are omitted
        }

def init_db(app):
    """
    Initialize database tables within the Flask app context.
    Creates all tables defined in the models if they don't already exist.
    """
    with app.app_context():
        try:
            db.create_all()
        except Exception as e:
            # IMPORTANT: Log this error instead of just printing in a production environment
            print(f"FATAL ERROR initializing database tables: {str(e)}")