"""
API Routes for Student Performance Predictor
Defines endpoints for single and batch student predictions.
"""

import logging
from datetime import datetime
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
# ⬇️ REQUIRED CHANGE HERE ⬇️
import traceback
# ⬆️ REQUIRED CHANGE HERE ⬆️

from models import db, Student, PredictionLog
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

        # Note: OpenRouter API uses the original, named 'data' for prompt context, 
        # which is correct, while HF API uses the ordered 'hf_features'.
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


        response = {
            "predicted_performance": predicted_score,
            "improvement_plan": improvement_plan
        }
        return jsonify({"data": response}), 200

    except Exception as e:
        # Log full stack trace
        logger.error("Error in /predict-single:\n" + traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@api_bp.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Endpoint for batch student performance predictions.
    """
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify(format_error_response("No file provided")), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify(format_error_response("No file selected")), 400
        
        # Validate file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            return jsonify(format_error_response(f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB")), 400
        
        # Parse file
        records, parse_error = FileParser.parse_file(file)
        
        if parse_error:
            return jsonify(format_error_response("File parsing failed", parse_error)), 400
        
        if not records:
            return jsonify(format_error_response("No valid records found in file")), 400
        
        logger.info(f"Batch processing started. Total records: {len(records)}")
        
        # Create prediction log entry
        pred_log = PredictionLog(
            prediction_type='batch',
            input_data={'total_records': len(records)},
            status='pending'
        )
        db.session.add(pred_log)
        db.session.commit()
        
        # Get existing records for duplicate checking
        existing_students = db.session.query(Student).all()
        existing_records = [{'name': s.name, 'year': s.year, 'semester': s.semester} for s in existing_students]
        
        # Filter duplicates
        unique_records, duplicate_records = DuplicateChecker.filter_duplicates(records, existing_records)
        
        logger.info(f"Unique records: {len(unique_records)}, Duplicates: {len(duplicate_records)}")
        
        # Process predictions
        predictions = []
        errors = []
        successful_count = 0
        
        for idx, record in enumerate(unique_records):
            try:
                # 1. Map and validate the input data order for HF
                hf_features, validation_error = prepare_hf_features(record['features'])
                if validation_error:
                    raise ValueError(validation_error)
                
                # 2. Get prediction
                predicted_score, hf_error = hf_api.predict(hf_features)
                
                if hf_error:
                    errors.append({
                        'record_index': idx,
                        'name': record['name'],
                        'error': f"Prediction failed: {hf_error}"
                    })
                    continue
                
                # 3. Generate improvement plan
                improvement_plan, or_error = openrouter_api.generate_improvement_plan(
                    record['name'],
                    record['year'],
                    record['semester'],
                    record['features'],
                    predicted_score
                )
                
                if or_error:
                    logger.warning(f"Plan generation failed for {record['name']}: {or_error}")
                    improvement_plan = "Plan generation temporarily unavailable."
                
                # 4. Store in database
                student = Student(
                    name=record['name'],
                    year=record['year'],
                    semester=record['semester'],
                    features=record['features'],
                    predicted_performance=predicted_score,
                    improvement_plan=improvement_plan
                )
                db.session.add(student)
                db.session.flush()  # Get the ID without committing
                
                predictions.append({
                    'student_id': student.id,
                    'name': student.name,
                    'year': student.year,
                    'semester': student.semester,
                    'predicted_performance': predicted_score,
                    'improvement_plan': improvement_plan
                })
                
                successful_count += 1
                
            except Exception as e:
                logger.error(f"Error processing record {idx}: {str(e)}")
                errors.append({
                    'record_index': idx,
                    'name': record.get('name', 'Unknown'),
                    'error': str(e)
                })
        
        # Commit all changes
        db.session.commit()
        
        # Update prediction log
        pred_log.predicted_value = successful_count
        pred_log.status = 'success' if successful_count > 0 else 'failed'
        db.session.commit()
        
        logger.info(f"Batch processing completed. Successful: {successful_count}, Failed: {len(errors)}")
        
        # Return response
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
        return jsonify(format_error_response("Internal server error", str(e))), 500


@api_bp.route('/history', methods=['GET'])
def get_history():
    """
    Endpoint to retrieve prediction history.
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Validate parameters
        limit = min(limit, 100)  # Max 100 records per request
        offset = max(offset, 0)
        
        # Query database
        students = db.session.query(Student).order_by(Student.created_at.desc()).limit(limit).offset(offset).all()
        total = db.session.query(Student).count()
        
        # Format response
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
    Endpoint to retrieve a specific student's prediction data.
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
            'download_url': f'/api/download/{pdf_filename}' # Use /api prefix as it's registered under it
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
        from flask import send_file
        import os
        
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