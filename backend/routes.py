API Routes for Student Performance Predictor
Defines endpoints for single and batch student predictions.
"""

import logging
from datetime import datetime
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import traceback

from models import db, Student, PredictionLog # Ensure Student is imported here
from utils import FileParser, DataValidator, DuplicateChecker, format_error_response, format_success_response
from external_apis import HuggingFaceAPI, OpenRouterAPI
from pdf_generator import ImprovementPlanPDFGenerator

logger = logging.getLogger(__name__)

# Create Blueprint for API routes
api_bp = Blueprint('api', __name__)

# Initialize external API clients
hf_api = HuggingFaceAPI()
openrouter_api = OpenRouterAPI()
pdf_generator = ImprovementPlanPDFGenerator()

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}


# 🌟 CRITICAL MAPPING FOR HUGGING FACE API 🌟
# ... (HF_INPUT_ORDER dictionary remains unchanged)
HF_INPUT_ORDER = {
    0: 'marital_status',
    1: 'application_mode',
    2: 'application_order',
    3: 'course',
    4: 'daytime_evening_attendance',
    5: 'previous_qualification',
    6: 'previous_qualification_grade',
    7: 'nationality',
    8: 'mothers_qualification',
    9: 'fathers_qualification',
    10: 'mothers_occupation',
    11: 'fathers_occupation',
    12: 'admission_grade',
    13: 'displaced',
    14: 'educational_special_needs',
    15: 'debtor',
    16: 'tuition_fees_up_to_date',
    17: 'gender',
    18: 'scholarship_holder',
    19: 'age_at_enrollment',
    20: 'international',
    21: 'curricular_units_1st_sem_credited',
    22: 'curricular_units_1st_sem_enrolled',
    23: 'curricular_units_1st_sem_evaluations',
    24: 'curricular_units_1st_sem_approved',
    25: 'curricular_units_1st_sem_grade',
    26: 'curricular_units_1st_sem_without_evaluations',
    27: 'curricular_units_2nd_sem_credited',
    28: 'curricular_units_2nd_sem_enrolled',
    29: 'curricular_units_2nd_sem_evaluations',
    30: 'curricular_units_2nd_sem_approved',
    31: 'curricular_units_2nd_sem_grade',
    32: 'curricular_units_2nd_sem_without_evaluations',
    33: 'unemployment_rate',
    34: 'inflation_rate',
    35: 'gdp'
}
# ----------------------------------------


def prepare_hf_features(input_data: dict) -> tuple[dict, str | None]:
    """
    Ensures input data is correctly ordered and converted to float for the HF model.
    Returns: (ordered_dict_of_36_features, error_message)
    """
    ordered_features = {}
    for i in range(36):
        key = HF_INPUT_ORDER.get(i)
        if not key:
            return {}, f"Internal error: Missing mapping for input parameter {i}."
        
        value = input_data.get(key)
        if value is None:
            # Check for a different casing/naming convention fallback if needed
            value = input_data.get(key.replace('_', '')) 

        if value is None:
            return {}, f"Missing required feature: {key}"

        try:
            # All 36 inputs must be floats or integers (handled as floats in Python)
            ordered_features[key] = float(value)
        except (ValueError, TypeError):
            return {}, f"Invalid data type for feature {key}. Expected numeric, got {value}."
    
    return ordered_features, None


@api_bp.route('/predict-single', methods=['POST'])
def predict_single():
    try:
        data = request.get_json()
        logger.info("Single prediction payload: %s", data)

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # 1. Map and validate the input data order
        hf_features, validation_error = prepare_hf_features(data)
        if validation_error:
            return jsonify({"error": validation_error}), 400

        # 2. Call your prediction function (using the correctly ordered features)
        predicted_score, error = hf_api.predict(hf_features)
        if error:
            return jsonify({"error": error}), 500

        # Note: OpenRouter API uses the original, named 'data' for prompt context
        improvement_plan, plan_error = openrouter_api.generate_improvement_plan(
            student_name=data.get('student_name', 'Unknown'),
            year=data.get('year', 1),
            semester=data.get('semester', 1),
            features=data, # Use the original named data for the LLM prompt
            predicted_score=predicted_score
        )
        
        if plan_error:
            # Handle plan generation failure gracefully
            logger.warning(f"Plan generation failed for single prediction: {plan_error}")
            improvement_plan = "Improvement plan generation temporarily unavailable."

        # 🌟 FIX 3: Store prediction in database 🌟
        student = Student(
            name=data.get('student_name', 'Single Prediction'),
            year=data.get('year', 1),
            semester=data.get('semester', 1),
            features=data,
            predicted_performance=predicted_score, # Categorical string
            improvement_plan=improvement_plan
        )
        db.session.add(student)
        db.session.commit() # Commit the transaction to save the record!

        response = {
            "predicted_performance": predicted_score,
            "improvement_plan": improvement_plan,
            "student_id": student.id # Return ID for potential history linking
        }
        return jsonify({"data": response}), 200

    except Exception as e:
        # Log full stack trace
        logger.error("Error in /predict-single:\n" + traceback.format_exc())
        # Rollback the session if an error occurred before commit
        db.session.rollback() 
        return jsonify({"error": str(e)}), 500


@api_bp.route('/predict-batch', methods=['POST'])
def predict_batch():
# ... (rest of the predict_batch function remains unchanged)
# ...

@api_bp.route('/history', methods=['GET'])
def get_history():
# ... (rest of the get_history function remains unchanged)
# ...

@api_bp.route('/student/<int:student_id>', methods=['GET'])
def get_student(student_id):
# ... (rest of the get_student function remains unchanged)
# ...

@api_bp.route('/student/<int:student_id>/pdf', methods=['GET'])
def download_student_pdf(student_id):
# ... (rest of the download_student_pdf function remains unchanged)
# ...

@api_bp.route('/download/<filename>', methods=['GET'])
def download_file(filename):
# ... (rest of the download_file function remains unchanged)
# ...