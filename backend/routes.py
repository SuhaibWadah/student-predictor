"""
API Routes for Student Performance Predictor
Defines endpoints for single and batch student predictions, using the consolidated 
Student model for logging.
"""

import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, send_from_directory # ✨ Changed send_file to send_from_directory
from werkzeug.utils import secure_filename
import traceback
import pandas as pd 
import os
from typing import Any

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
pdf_generator = ImprovementPlanPDFGenerator() # Initialize here, output_dir='pdfs'

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}


# 🌟 CRITICAL MAPPING FOR HUGGING FACE API 🌟
# (The HF_INPUT_ORDER dictionary remains unchanged as it maps to the 36 features)
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
    Imputes 'gdp' if missing/NaN and ensures the input_data dictionary is safe 
    for JSON serialization by replacing Python NaNs with floats.
    
    Returns: (ordered_dict_of_36_features, error_message)
    """
    ordered_features = {}
    
    # Define the default GDP value
    DEFAULT_GDP = 17.4 # Placeholder value
    
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
        # This replaces the original file's value with the cleaned float/imputed value
        input_data[key] = ordered_features[key]
    
    return ordered_features, None


@api_bp.route('/predict-single', methods=['POST'])
def predict_single():
    try:
        data = request.get_json()
        logger.info("Single prediction payload received.")

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # --- ✨ START: New name/year/major extraction and validation ---
        student_name = data.get('name', 'Unknown')
        student_year = data.get('year') # Use 'year' field directly from front-end
        student_major = data.get('major', 'N/A')
        
        if not student_year or student_year not in [1, 2, 3, 4]:
             return jsonify({"error": "Invalid or missing 'year'. Must be 1, 2, 3, or 4."}), 400
        
        # Ensure the name stored is the user-friendly name
        data_to_store_name = student_name 
        # --- END: New name/year/major extraction ---


        # 1. Map and validate the input data order (gdp is corrected inside this call)
        # Note: 'data' is modified in place to clean the feature values
        hf_features, validation_error = prepare_hf_features(data)
        if validation_error:
            return jsonify({"error": validation_error}), 400
        
        # 2. Call your prediction function
        predicted_score, error = hf_api.predict(hf_features)
        
        if error:
            return jsonify({"error": error}), 500

        # 3. Generate improvement plan
        # NOTE: Removed 'semester' usage and confidence/version as they are no longer stored
        improvement_plan, plan_error = openrouter_api.generate_improvement_plan(
            student_name=student_name,
            year=student_year,
            features=data, # Use the cleaned 'data'
            predicted_score=predicted_score
        )
        
        if plan_error:
            logger.warning(f"Plan generation failed for single prediction: {plan_error}")
            improvement_plan = "Improvement plan generation temporarily unavailable."

        # 4. Store prediction in database
        student = Student(
            name=data_to_store_name, # Use the cleaned student name
            year=student_year,
            major=student_major, # Store major
            features=data, 
            predicted_performance=predicted_score, 
            improvement_plan=improvement_plan,
            # ❌ REMOVED: prediction_model_version, confidence_score, semester
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
        
        # Parse file (records contains name, year, major, and features)
        records, parse_error = FileParser.parse_file(file)
        
        if parse_error:
            return jsonify(format_error_response("File parsing failed", parse_error)), 400
        
        if not records:
            return jsonify(format_error_response("No valid records found in file")), 400
        
        logger.info(f"Batch processing started. Total records: {len(records)}")
        
        # Get existing records for duplicate checking (checking name and year)
        existing_students = db.session.query(Student).all()
        existing_records = [{'name': s.name, 'study_year': s.year, 'major': s.major} for s in existing_students]
        
        # Filter duplicates
        unique_records, duplicate_records = DuplicateChecker.filter_duplicates(records, existing_records)
        
        logger.info(f"Unique records: {len(unique_records)}, Duplicates: {len(duplicate_records)}")
        
        # Process predictions
        predictions = []
        errors = []
        successful_count = 0
        
        for idx, record in enumerate(unique_records):
            student_name = record.get('name')
            student_year = record.get('year') # Uses 'year' from utils.py's parsing
            student_major = record.get('major')
            
            try:
                # The 'record' dictionary contains the name/year/major AND the features
                # CRITICAL: record is modified in place to fix 'gdp' (NaN -> 17.4)
                hf_features, validation_error = prepare_hf_features(record)
                
                if validation_error:
                    raise ValueError(f"Record {student_name}: {validation_error}")
                
                # 2. Get prediction
                predicted_score, hf_error = hf_api.predict(hf_features)
                
                if hf_error:
                    errors.append({'record_index': idx, 'name': student_name, 'error': f"Prediction failed: {hf_error}"})
                    continue
                
                # 3. Generate improvement plan
                improvement_plan, or_error = openrouter_api.generate_improvement_plan(
                    student_name,
                    student_year,
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
                    major=student_major, # Store major
                    features=record, # record is now safe for JSON storage
                    predicted_performance=predicted_score,
                    improvement_plan=improvement_plan,
                    # ❌ REMOVED: prediction_model_version, confidence_score, semester
                )
                db.session.add(student)
                db.session.flush()  # Use flush to get the ID
                
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
    
    This function generates the PDF and returns a URL to the file, which the front-end
    can then navigate to or fetch for the actual file download.
    """
    try:
        student = db.session.query(Student).filter_by(id=student_id).first()
        
        if not student:
            return jsonify(format_error_response("Student not found")), 404
        
        # Generate PDF
        pdf_filename, error = pdf_generator.generate_improvement_plan_pdf(
            student.name,
            student.year,
            # ✨ Ensure parameters match the updated pdf_generator signature
            predicted_score=student.predicted_performance, 
            improvement_plan=student.improvement_plan,
            features=student.features if isinstance(student.features, dict) else {}
        )
        
        if error:
            logger.error(f"PDF generation failed: {error}")
            return jsonify(format_error_response("PDF generation failed", error)), 500
        
        # Return file path for download
        # CRITICAL: The download URL must use the blueprint prefix '/api'
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
    This route streams the file directly to the client.
    """
    try:
        # 1. Use secure_filename to clean the input path component
        secure_name = secure_filename(filename)
        if secure_name != filename:
            logger.warning(f"Rejected insecure filename attempt: {filename}")
            return jsonify(format_error_response("Invalid filename format")), 400
        
        # 2. Get the *absolute* path to the PDF directory from the generator instance
        # This is typically 'pdfs' relative to the app's root
        pdf_directory = pdf_generator.output_dir
        
        # 3. Use Flask's send_from_directory for secure and robust file serving
        return send_from_directory(
            directory=pdf_directory,
            path=secure_name,
            as_attachment=True,
            download_name=secure_name
        )
        
    # CRITICAL: If send_from_directory fails (e.g., file not found), handle it gracefully
    except FileNotFoundError:
        logger.error(f"Download file not found: {filename}")
        return jsonify(format_error_response("File not found", f"The requested file '{filename}' does not exist.")), 404
    except Exception as e:
        # The generic exception handler that was causing the cryptic JSON response
        logger.error(f"Error in download_file:\n" + traceback.format_exc())
        return jsonify(format_error_response("Internal server error", str(e))), 500