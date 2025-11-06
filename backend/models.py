"""
Database Models for Student Performance Predictor
Defines SQLAlchemy ORM models for storing student data, predictions, and improvement plans.
"""

import json
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
# NOTE: Using 'sqlalchemy.dialects.sqlite.JSON' is fine for SQLite, 
# but for PostgreSQL, Flask-SQLAlchemy handles JSON automatically 
# if the environment is set up correctly.

# Initialize SQLAlchemy database instance
db = SQLAlchemy()

class Student(db.Model):
    """
    Student model to store student information and their prediction records.
    """
    
    __tablename__ = 'students'
    
    # Primary key
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    
    # Student information
    name = db.Column(db.String(255), nullable=False, index=True)
    year = db.Column(db.Integer, nullable=False)
    semester = db.Column(db.Integer, nullable=False)
    
    # Student features as JSON (flexible schema for various feature types)
    features = db.Column(db.JSON, nullable=False) # Changed JSON import for clarity/compatibility
    
    # Prediction results
    # 🌟 FIX: Change from db.Float to db.String to store the categorical result 🌟
    predicted_performance = db.Column(db.String(50), nullable=True) 
    improvement_plan = db.Column(db.Text, nullable=True)
    
    # Metadata
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        """String representation of Student object"""
        return f'<Student {self.id}: {self.name} (Year {self.year}, Sem {self.semester})>'
    
    def to_dict(self):
        """
        Convert Student object to dictionary for JSON serialization.
        """
        return {
            'id': self.id,
            'name': self.name,
            'year': self.year,
            'semester': self.semester,
            'features': self.features if isinstance(self.features, dict) else json.loads(self.features),
            'predicted_performance': self.predicted_performance,
            'improvement_plan': self.improvement_plan,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class PredictionLog(db.Model):
    """
    Prediction log model to track all prediction requests for auditing and analytics.
    """
    
    __tablename__ = 'prediction_logs'
    
    # Primary key
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    
    # Foreign key reference
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=True, index=True)
    
    # Prediction details
    prediction_type = db.Column(db.String(50), nullable=False)  # 'single' or 'batch'
    input_data = db.Column(db.JSON, nullable=False) # Changed to use db.JSON
    
    # 🌟 FIX: Change from db.Float to db.String to match the Student model 🌟
    predicted_value = db.Column(db.String(50), nullable=True)
    confidence_score = db.Column(db.Float, nullable=True)
    
    # Performance metrics
    api_response_time = db.Column(db.Integer, nullable=True)  # milliseconds
    
    # Status tracking
    status = db.Column(db.String(50), nullable=False, default='pending')  # 'success', 'failed', 'partial'
    error_message = db.Column(db.Text, nullable=True)
    
    # Metadata
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        """String representation of PredictionLog object"""
        return f'<PredictionLog {self.id}: {self.prediction_type} - {self.status}>'
    
    def to_dict(self):
        """
        Convert PredictionLog object to dictionary for JSON serialization.
        """
        return {
            'id': self.id,
            'student_id': self.student_id,
            'prediction_type': self.prediction_type,
            'input_data': self.input_data if isinstance(self.input_data, dict) else json.loads(self.input_data),
            'predicted_value': self.predicted_value,
            'confidence_score': self.confidence_score,
            'api_response_time': self.api_response_time,
            'status': self.status,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat()
        }


def init_db():
    """
    Initialize database tables.
    Creates all tables defined in the models if they don't already exist.
    Should be called once when the application starts.
    """
    try:
        db.create_all()
    except Exception as e:
        print(f"Error initializing database: {str(e)}")