"""
API Routes for Student Performance Predictor
Defines endpoints for single and batch student predictions.
"""

import logging
from datetime import datetime
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import traceback
import pandas as pd 

from models import db, Student # Removed PredictionLog
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
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}


# The HF_INPUT_ORDER is crucial for the model and remains correct.
HF_INPUT_ORDER = {
    0: 'marital_status', 1: 'application_mode', 2: 'application_order', 3: 'course',
    4: 'daytime_evening_attendance', 5: 'previous_qualification', 6: 'previous_qualification_grade', 7: 'nacionality',
    8: 'mothers_qualification', 9: 'fathers_qualification', 10: 'mothers_occupation', 11: 'fathers_occupation',
    12: 'admission_grade', 13: 'displaced', 14: 'educational_special_needs', 15: 'debtor',
    16: 'tuition_fees_up_to_date', 17: 'gender', 18: 'scholarship_holder', 19: 'age_at_enrollment',
    20: 'international', 21: 'curricular_units_1st_sem_credited', 22: 'curricular_units_1st_sem_enrolled',
    23: 'curricular_units_1st_sem_evaluations', 24: 'curricular_units_1st_sem_approved', 25: 'curricular_units_1st_sem_grade',
    26: 'curricular_units_1st_sem_without_evaluations', 27: 'curricular_units_2nd_sem_credited', 28: 'curricular_units_2nd_sem_enrolled',
    29: 'curricular_units_2nd_sem_evaluations', 30: 'curricular_units_2nd_sem_approved', 31: 'curricular_units_2nd_sem_grade',
    32: 'curricular_units_2nd_sem_without_evaluations', 33: 'unemployment_rate', 34: 'inflation_rate', 35: 'gdp'
}


def prepare_hf_features(input_data: dict) -> tuple[dict, str | None]:
    """
    Ensures input data is correctly ordered and converted to float for the HF model.
    Imputes 'gdp' if missing/NaN and replaces Python 'NaN' with a valid float 
    in the original 'input_data' dictionary to prevent downstream errors.
    
    Returns: (ordered_dict_of_36_features, error_message)
    """
    ordered_features = {}
    
    DEFAULT_GDP = 17.4

    for i in range(36):
        key = HF_INPUT_ORDER.get(i)
        if not key:
            return {}, f"Internal error: Missing mapping for input parameter {i}."
        
        value = input_data.get(key)
        
        is_null_or_nan = pd.isna(value)

        if is_null_or_nan:
            if key == 'gdp':
                value = DEFAULT_GDP
                logger.warning(f"Feature '{key}' was missing/null. Imputing with default value: {DEFAULT_GDP}")
            else:
                 return {}, f"Missing required feature: {key}"
        
        try:
            float_value = float(value)
            ordered_features[key] = float_value
        except (ValueError, TypeError):
            return {}, f"Invalid data type for feature {key}. Expected numeric, got {value!r}."

        if key in input_data:
             input_data[key] = ordered_features[key]
    
    return ordered_features, None


def _extract_all_db_fields(data: dict, predicted_score: str = None, improvement_plan: str = None) -> dict:
    """Helper to extract all fields necessary for the wide-format Student model."""
    
    # Map the core features from the data dictionary (must match model columns)
    db_fields = {key: data.get(key) for key in HF_INPUT_ORDER.values()}
    
    # Add new identifier fields (must match model columns)
    db_fields['name'] = data.get('name', data.get('student_name', 'Single Prediction'))
    db_fields['major'] = data.get('major', None)
    db_fields['study_year'] = data.get('study_year', None)
    
    # Add prediction results
    if predicted_score:
        db_fields['predicted_performance'] = str(predicted_score).replace("Predicted class: ", "")
    if improvement_plan:
        db_fields['improvement_plan'] = improvement_plan
    
    return db_fields


@api_bp.route('/predict-single', methods=['POST'])
def predict_single():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # The input data must contain the 36 features + name + optional identifiers (major, study_year)
        hf_features, validation_error = prepare_hf_features(data)
        if validation_error:
            return jsonify({"error": validation_error}), 400

        # 1. Prediction
        predicted_score, error = hf_api.predict(hf_features)
        if error:
            return jsonify({"error": error}), 500

        # 2. Plan Generation (using study_year instead of year/semester)
        study_year = data.get('study_year', data.get('year', 'Unknown')) # Fallback for prompt
        improvement_plan, plan_error = openrouter_api.generate_improvement_plan(
            student_name=data.get('name', data.get('student_name', 'Unknown')),
            study_year=study_year,
            features=data,
            predicted_score=predicted_score
        )
        
        if plan_error:
            logger.warning(f"Plan generation failed for single prediction: {plan_error}")
            improvement_plan = "Improvement plan generation temporarily unavailable."

        # 3. Store prediction in database (Wide-Format)
        db_fields = _extract_all_db_fields(
            data,
            predicted_score=predicted_score,
            improvement_plan=improvement_plan
        )
        
        student = Student(**db_fields)
        
        db.session.add(student)
        db.session.commit()
        
        response = {
            "predicted_performance": student.predicted_performance,
            "improvement_plan": student.improvement_plan,
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
    """
    try:
        if 'file' not in request.files:
            return jsonify(format_error_response("No file provided")), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify(format_error_response("No file selected")), 400
        
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify(format_error_response(f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB")), 400
        
        # Parse file (records are now FLAT dictionaries of all 36 features + identifiers)
        records, parse_error = FileParser.parse_file(file)
        
        if parse_error:
            return jsonify(format_error_response("File parsing failed", parse_error)), 400
        
        if not records:
            return jsonify(format_error_response("No valid records found in file")), 400
        
        logger.info(f"Batch processing started. Total records: {len(records)}")
        
        # NOTE: PredictionLog creation removed as requested.
        
        # Get existing records for duplicate checking (only need name, study_year for uniqueness)
        existing_students = db.session.query(Student).all()
        existing_records = [{'name': s.name, 'study_year': s.study_year} for s in existing_students]
        
        # Filter duplicates based on name and study_year
        unique_records, duplicate_records = DuplicateChecker.filter_duplicates(
            [{'name': r['name'], 'study_year': r.get('study_year')} for r in records], 
            existing_records
        )
        
        logger.info(f"Unique records: {len(unique_records)}, Duplicates: {len(duplicate_records)}")
        
        predictions = []
        errors = []
        successful_count = 0
        
        for idx, record in enumerate(unique_records):
            try:
                # 1. Prepare Features (record is already a flat dictionary containing all features)
                # prepare_hf_features extracts the 36 features for the model and handles GDP imputation.
                hf_features, validation_error = prepare_hf_features(record)
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
                study_year = record.get('study_year', 'Unknown')
                improvement_plan, or_error = openrouter_api.generate_improvement_plan(
                    record['name'],
                    study_year,
                    record, # Pass the flat record which contains all features
                    predicted_score
                )
                
                if or_error:
                    logger.warning(f"Plan generation failed for {record['name']}: {or_error}")
                    improvement_plan = "Plan generation temporarily unavailable."
                
                # 4. Store in database (Wide-Format)
                db_fields = _extract_all_db_fields(
                    record,
                    predicted_score=predicted_score,
                    improvement_plan=improvement_plan
                )
                
                student = Student(**db_fields)
                db.session.add(student)
                db.session.flush()
                
                predictions.append({
                    'student_id': student.id,
                    'name': student.name,
                    'study_year': student.study_year,
                    'predicted_performance': student.predicted_performance,
                    'improvement_plan': student.improvement_plan
                })
                
                successful_count += 1
                
            except Exception as e:
                logger.error(f"Error processing record {idx}: {str(e)}")
                errors.append({
                    'record_index': idx,
                    'name': record.get('name', 'Unknown'),
                    'error': str(e)
                })
        
        db.session.commit()
        
        # NOTE: PredictionLog update removed.
        
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
    Endpoint to retrieve prediction history.
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        limit = min(limit, 100)
        offset = max(offset, 0)
        
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
        
        # NOTE: PDF generator needs a dictionary of features for context, so we pass student.to_dict()
        pdf_filename, error = pdf_generator.generate_improvement_plan_pdf(
            student.name,
            student.study_year,
            None, # Removed semester arg
            student.predicted_performance,
            student.improvement_plan,
            student.to_dict() # Pass the full dictionary of features
        )
        
        if error:
            logger.error(f"PDF generation failed: {error}")
            return jsonify(format_error_response("PDF generation failed", error)), 500
        
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
        from flask import send_file
        import os
        
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify(format_error_response("Invalid filename")), 400
        
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