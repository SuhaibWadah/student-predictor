"""
API Routes for Student Performance Predictor
Defines endpoints for single and batch student predictions, using the consolidated 
Student model for logging.
"""

import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
import traceback
import pandas as pd 
import os
from typing import Any

# 🌟 CRITICAL FIX: Only import Student, remove PredictionLog
from models import db, Student 
from utils import FileParser, DataValidator, DuplicateChecker, format_error_response, format_success_response
# Placeholder imports for external services (assuming they exist in your project)
from external_apis import HuggingFaceAPI, OpenRouterAPI
from pdf_generator import ImprovementPlanPDFGenerator

logger = logging.getLogger(__name__)

# Create Blueprint for API routes
api_bp = Blueprint('api', __name__)

# Initialize external API clients (placeholders)
hf_api = HuggingFaceAPI()
openrouter_api = OpenRouterAPI()
pdf_generator = ImprovementPlanPDFGenerator()

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}


# 🌟 CRITICAL MAPPING FOR HUGGING FACE API 🌟
# This dictionary maps the user-friendly keys (from your JSON/DB) to the 
# POSITIONAL ORDER (param_0 to param_35) required by your Gradio Space model.
# NOTE: YOU MUST verify this order against your actual Gradio Space inputs!
HF_INPUT_ORDER = {
    0: 'marital_status',
    1: 'application_mode',
    2: 'application_order',
    3: 'course',
    4: 'daytime_evening_attendance',
    5: 'previous_qualification',
    6: 'previous_qualification_grade',
    7: 'nationality', # CORRECTED: Mapped to the correct key in data
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
    Imputes 'gdp' if missing/NaN and ensures the input_data dictionary is safe 
    for JSON serialization by replacing Python NaNs with floats.
    
    Returns: (ordered_dict_of_36_features, error_message)
    """
    ordered_features = {}
    
    # Define the default GDP value
    DEFAULT_GDP = 17.4 # Placeholder value
    # Placeholder values for prediction audit (replace with actual model output)
    GLOBAL_PREDICTED_SCORE = "Success"
    GLOBAL_CONFIDENCE_SCORE = 0.95
    
    for i in range(36):
        key = HF_INPUT_ORDER.get(i)
        if not key:
            return {}, f"Internal error: Missing mapping for input parameter {i}."
        
        value = input_data.get(key)
        
        is_null_or_nan = pd.isna(value)

        # Imputation/Fallback Logic
        if is_null_or_nan:
            if key == 'gdp':
                value = DEFAULT_GDP
                logger.warning(f"Feature '{key}' was missing/null. Imputing with default value: {DEFAULT_GDP}")
            else:
                 # Non-GDP critical features must be present
                 return {}, f"Missing required feature: {key}"
        
        # Ensure final value is a float for the model
        try:
            float_value = float(value)
            ordered_features[key] = float_value
        except (ValueError, TypeError):
            return {}, f"Invalid data type for feature {key}. Expected numeric, got {value!r}."

        # CRITICAL: Update the original input_data dictionary for database saving
        if key in input_data:
             input_data[key] = ordered_features[key]
    
    return ordered_features, None


@api_bp.route('/predict-single', methods=['POST'])
def predict_single():
    try:
        data = request.get_json()
        logger.info("Single prediction payload received.")

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # 1. Map and validate the input data order (gdp is corrected inside this call)
        hf_features, validation_error = prepare_hf_features(data)
        if validation_error:
            return jsonify({"error": validation_error}), 400
        
        start_time = datetime.utcnow() # Track API time
        
        # 2. Call your prediction function
        # ASSUMPTION: hf_api.predict returns (predicted_class_string, confidence_score_float, error)
        # If your API only returns one score, you must adapt this.
        # For simplicity, we assume it returns the predicted class string (e.g., 'Success').
        predicted_score, error = hf_api.predict(hf_features)
        
        if error:
            return jsonify({"error": error}), 500

        # Placeholder values for audit fields
        confidence_score = data.get('confidence_score', 0.95) # Replace with actual model output
        model_version = "HF_SPACE_V1"

        # 3. Generate improvement plan
        improvement_plan, plan_error = openrouter_api.generate_improvement_plan(
            student_name=data.get('name', 'Unknown'),
            year=data.get('study_year', 1), # Using study_year from the data
            semester=data.get('semester', 1), # Assuming a default semester value if not provided
            features=data, 
            predicted_score=predicted_score
        )
        
        if plan_error:
            logger.warning(f"Plan generation failed for single prediction: {plan_error}")
            improvement_plan = "Improvement plan generation temporarily unavailable."

        # 4. Store prediction in database
        student = Student(
            name=data.get('name', 'Single Prediction'),
            year=data.get('study_year', 1),
            semester=data.get('semester', 1), # Use data from request or default
            features=data, # Use 'data' which now has the corrected 'gdp'
            predicted_performance=predicted_score, 
            improvement_plan=improvement_plan,
            prediction_model_version=model_version,
            confidence_score=confidence_score
        )
        db.session.add(student)
        db.session.commit() 
        
        response = {
            "predicted_performance": predicted_score,
            "improvement_plan": improvement_plan,
            "student_id": student.id
        }
        return jsonify({"data": response}), 200

    except Exception as e:
        logger.error("Error in /predict-single:\n" + traceback.format_exc())
        db.session.rollback() 
        return jsonify({"error": str(e)}), 500


@api_bp.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Endpoint for batch student performance predictions.
    Saves results directly to the 'students' table.
    """
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify(format_error_response("No file provided or selected")), 400
        
        file = request.files['file']
        
        # ... (File size check remains the same) ...
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0) 
        if file_size > MAX_FILE_SIZE:
            return jsonify(format_error_response(f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB")), 400
        
        # Parse file
        records, parse_error = FileParser.parse_file(file)
        
        if parse_error:
            return jsonify(format_error_response("File parsing failed", parse_error)), 400
        
        if not records:
            return jsonify(format_error_response("No valid records found in file")), 400
        
        logger.info(f"Batch processing started. Total records: {len(records)}")
        
        # NO PredictionLog created here (removed)
        
        # Get existing records for duplicate checking
        # Assuming the 'study_year' from the file maps to the 'year' field in the DB
        existing_students = db.session.query(Student).all()
        existing_records = [{'name': s.name, 'study_year': s.year, 'major': s.major} for s in existing_students]
        
        # Filter duplicates
        # NOTE: DuplicateChecker assumes 'study_year' is used, aligning with your original code's requirement.
        unique_records, duplicate_records = DuplicateChecker.filter_duplicates(records, existing_records)
        
        logger.info(f"Unique records: {len(unique_records)}, Duplicates: {len(duplicate_records)}")
        
        # Process predictions
        predictions = []
        errors = []
        successful_count = 0
        
        for idx, record in enumerate(unique_records):
            try:
                # IMPORTANT: Map 'study_year' from record to 'year' in the DB model
                student_name = record.get('name')
                student_year = record.get('study_year')
                student_major = record.get('major')
                
                # The remaining keys are in record, we map them as features
                # CRITICAL: record is modified in place to fix 'gdp' (NaN -> 17.4)
                hf_features, validation_error = prepare_hf_features(record) # Pass the whole record dictionary
                
                if validation_error:
                    raise ValueError(f"Record {student_name}: {validation_error}")
                
                # 2. Get prediction
                predicted_score, hf_error = hf_api.predict(hf_features)
                
                if hf_error:
                    errors.append({'record_index': idx, 'name': student_name, 'error': f"Prediction failed: {hf_error}"})
                    continue
                
                # Placeholder values for audit fields
                confidence_score = record.get('confidence_score', 0.90) # Replace with actual model output
                model_version = "BATCH_RUN_V1"

                # 3. Generate improvement plan
                improvement_plan, or_error = openrouter_api.generate_improvement_plan(
                    student_name,
                    student_year,
                    # Assuming a default semester value if not in the file
                    record.get('semester', 1), 
                    record, # Use the corrected features dictionary
                    predicted_score
                )
                
                if or_error:
                    logger.warning(f"Plan generation failed for {student_name}: {or_error}")
                    improvement_plan = "Plan generation temporarily unavailable."
                
                # 4. Store in database
                student = Student(
                    name=student_name,
                    year=student_year,
                    semester=record.get('semester', 1), # Assuming semester is 1 if not provided
                    features=record, # record is now safe for JSON storage
                    predicted_performance=predicted_score,
                    improvement_plan=improvement_plan,
                    prediction_model_version=model_version,
                    confidence_score=confidence_score
                )
                db.session.add(student)
                db.session.flush()  
                
                predictions.append({
                    'student_id': student.id,
                    'name': student.name,
                    'predicted_performance': predicted_score,
                })
                
                successful_count += 1
                
            except Exception as e:
                logger.error(f"Error processing record {idx}: {str(e)}")
                errors.append({'record_index': idx, 'name': student_name, 'error': str(e)})
        
        # Commit all successful records
        db.session.commit()
        
        # NO PredictionLog update here (removed)
        
        logger.info(f"Batch processing completed. Successful: {successful_count}, Failed: {len(errors)}")
        
        response_data = {
            'total_records': len(records),
            'successful': successful_count,
            'failed': len(errors),
            'duplicates': len(duplicate_records),
            'predictions': predictions,
            'errors': errors,
            'duplicate_names': [r['name'] for r in duplicate_records] if duplicate_records else []
        }
        
        return jsonify(format_success_response(response_data, "Batch processing completed")), 200
        
    except Exception as e:
        logger.error(f"Error in predict_batch: {str(e)}")
        db.session.rollback() 
        return jsonify(format_error_response("Internal server error", str(e))), 500


@api_bp.route('/history', methods=['GET'])
def get_history():
    """
    Endpoint to retrieve prediction history from the consolidated 'students' table.
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        limit = min(limit, 100)
        offset = max(offset, 0)
        
        # Query database - only Student model used
        students = db.session.query(Student).order_by(Student.created_at.desc()).limit(limit).offset(offset).all()
        total = db.session.query(Student).count()
        
        data = [student.to_dict() for student in students]
        
        return jsonify({
            'data': data,
            'total': total,
            'limit': limit,
            'offset': offset
        }), 200
        
    except Exception as e:
        logger.error(f"Error in get_history: {str(e)}")
        return jsonify(format_error_response("Internal server error", str(e))), 500


@api_bp.route('/student/<int:student_id>', methods=['GET'])
def get_student(student_id):
    """
    Endpoint to retrieve a specific student's prediction data from the Student table.
    """
    try:
        student = db.session.query(Student).filter_by(id=student_id).first()
        
        if not student:
            return jsonify(format_error_response("Student not found")), 404
        
        return jsonify(format_success_response(student.to_dict())), 200
        
    except Exception as e:
        logger.error(f"Error in get_student: {str(e)}")
        return jsonify(format_error_response("Internal server error", str(e))), 500


@api_bp.route('/student/<int:student_id>/pdf', methods=['GET'])
def download_student_pdf(student_id):
    """
    Endpoint to download a student's improvement plan as PDF.
    """
    try:
        student = db.session.query(Student).filter_by(id=student_id).first()
        
        if not student:
            return jsonify(format_error_response("Student not found")), 404
        
        # Generate PDF
        pdf_filename, error = pdf_generator.generate_improvement_plan_pdf(
            student.name,
            student.year,
            student.semester,
            student.predicted_performance,
            student.improvement_plan,
            student.features if isinstance(student.features, dict) else {}
        )
        
        if error:
            logger.error(f"PDF generation failed: {error}")
            return jsonify(format_error_response("PDF generation failed", error)), 500
        
        # Return file path for download
        return jsonify(format_success_response({
            'filename': pdf_filename,
            'download_url': f'/api/download/{pdf_filename}' 
        }, "PDF generated successfully")), 200
        
    except Exception as e:
        logger.error(f"Error in download_student_pdf: {str(e)}")
        return jsonify(format_error_response("Internal server error", str(e))), 500


@api_bp.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """
    Endpoint to download generated PDF files.
    """
    try:
        # Security: Validate filename to prevent directory traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify(format_error_response("Invalid filename")), 400
        
        # CRITICAL: This path must be correct for Render/local FS
        pdf_path = os.path.join(os.getcwd(), 'pdfs', filename) 
        
        if not os.path.exists(pdf_path):
            return jsonify(format_error_response("File not found")), 404
        
        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Error in download_file: {str(e)}")
        return jsonify(format_error_response("Internal server error", str(e))), 500