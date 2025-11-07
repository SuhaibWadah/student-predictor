"""
Database Models for Student Performance Predictor
Defines a wide SQLAlchemy ORM model (Student) to store all original features, 
new identifiers, and the prediction result in a single table for detailed research.
"""

import json
from datetime import date
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func

# Initialize SQLAlchemy database instance
db = SQLAlchemy()

# Define the data types for repeated columns
# Note: Most of your original features are categorical codes or numeric values. 
# They are best stored as Integer or Float for direct analysis.

class Student(db.Model):
    """
    Student model storing all 36 input features plus identifiers and the final prediction.
    Designed for comprehensive data logging and research.
    """
    
    __tablename__ = 'students'
    
    # ----------------------------------------
    # I. IDENTIFIERS AND METADATA (New/Required)
    # ----------------------------------------
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    
    # New Identifier Fields
    name = db.Column(db.String(255), nullable=False, index=True)
    major = db.Column(db.String(100), nullable=True) # Added Major
    study_year = db.Column(db.String(50), nullable=True) # Added Study Year (e.g., 'First', 'Second')
    
    # Prediction Result
    # Stores only the result class, e.g., 'Success', 'Failure', 'Dropout'.
    predicted_performance = db.Column(db.String(50), nullable=True) 
    improvement_plan = db.Column(db.Text, nullable=True) # Storing the full LLM plan
    
    # Metadata (Date only)
    created_at = db.Column(db.Date, nullable=False, default=date.today, index=True)
    
    # ----------------------------------------
    # II. CORE STUDENT FEATURES (36 Columns)
    # ----------------------------------------
    # NOTE: Underscores replace spaces/special characters for SQL column names.

    # Personal & Application
    marital_status = db.Column(db.Integer, nullable=True)
    application_mode = db.Column(db.Integer, nullable=True)
    application_order = db.Column(db.Integer, nullable=True)
    course = db.Column(db.Integer, nullable=True) # Corresponds to Course ID
    daytime_evening_attendance = db.Column(db.Integer, nullable=True)
    previous_qualification = db.Column(db.Integer, nullable=True)
    previous_qualification_grade = db.Column(db.Float, nullable=True)
    nacionality = db.Column(db.Integer, nullable=True)
    
    # Parents' Data
    mothers_qualification = db.Column(db.Integer, nullable=True)
    fathers_qualification = db.Column(db.Integer, nullable=True)
    mothers_occupation = db.Column(db.Integer, nullable=True)
    fathers_occupation = db.Column(db.Integer, nullable=True)
    admission_grade = db.Column(db.Float, nullable=True)
    
    # Status Indicators
    displaced = db.Column(db.Integer, nullable=True)
    educational_special_needs = db.Column(db.Integer, nullable=True)
    debtor = db.Column(db.Integer, nullable=True)
    tuition_fees_up_to_date = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.Integer, nullable=True)
    scholarship_holder = db.Column(db.Integer, nullable=True)
    age_at_enrollment = db.Column(db.Integer, nullable=True)
    international = db.Column(db.Integer, nullable=True)
    
    # 1st Semester Performance Metrics
    curricular_units_1st_sem_credited = db.Column(db.Integer, nullable=True)
    curricular_units_1st_sem_enrolled = db.Column(db.Integer, nullable=True)
    curricular_units_1st_sem_evaluations = db.Column(db.Integer, nullable=True)
    curricular_units_1st_sem_approved = db.Column(db.Integer, nullable=True)
    curricular_units_1st_sem_grade = db.Column(db.Float, nullable=True) # GPA/Average Grade
    curricular_units_1st_sem_without_evaluations = db.Column(db.Integer, nullable=True)
    
    # 2nd Semester Performance Metrics
    curricular_units_2nd_sem_credited = db.Column(db.Integer, nullable=True)
    curricular_units_2nd_sem_enrolled = db.Column(db.Integer, nullable=True)
    curricular_units_2nd_sem_evaluations = db.Column(db.Integer, nullable=True)
    curricular_units_2nd_sem_approved = db.Column(db.Integer, nullable=True)
    curricular_units_2nd_sem_grade = db.Column(db.Float, nullable=True) # GPA/Average Grade
    curricular_units_2nd_sem_without_evaluations = db.Column(db.Integer, nullable=True)
    
    # Macro-Economic Factors
    unemployment_rate = db.Column(db.Float, nullable=True)
    inflation_rate = db.Column(db.Float, nullable=True)
    gdp = db.Column(db.Float, nullable=True)
    
    # Target Variable (if collected later)
    target = db.Column(db.String(50), nullable=True) # E.g., 'Dropout', 'Success', 'Graduated'
    
    # ----------------------------------------

    def __repr__(self):
        """String representation of Student object"""
        return f'<Student {self.id}: {self.name} (Year {self.study_year}) - {self.predicted_performance}>'
    
    def to_dict(self):
        """
        Convert Student object to dictionary for JSON serialization.
        """
        # Automatically collect all attributes defined on the model
        data = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        
        # Ensure date is correctly formatted
        if 'created_at' in data and data['created_at']:
             data['created_at'] = data['created_at'].isoformat()
             
        return data

# Removed: PredictionLog class as requested

def init_db():
    """
    Initialize database tables.
    Creates all tables defined in the models if they don't already exist.
    """
    try:
        db.create_all()
        print("Database tables created successfully (Student table only - Wide Format).")
    except Exception as e:
        print(f"Error initializing database: {str(e)}")