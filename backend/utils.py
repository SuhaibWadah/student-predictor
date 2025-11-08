"""
Utility functions for Student Performance Predictor
Provides file parsing, data validation, and helper functions.
"""

import json
import pandas as pd
import logging
from io import BytesIO
from werkzeug.datastructures import FileStorage
from typing import Any
logger = logging.getLogger(__name__)

class FileParser:
    """
    Utility class for parsing various file formats (CSV, Excel).
    Supports batch student data processing.
    """
    
    SUPPORTED_FORMATS = {'.csv', '.xlsx', '.xls'}
    
    # Updated to reflect the wide format table structure
    REQUIRED_FIELDS = {
        'name', 'study_year', 'major', # New/renamed identifiers
        
        'marital_status', 'application_mode', 'application_order', 
        'course', 'daytime_evening_attendance', 'previous_qualification', 'previous_qualification_grade', 
        'nacionality', 'mothers_qualification', 'fathers_qualification', 'mothers_occupation', 
        'fathers_occupation', 'admission_grade', 'displaced', 'educational_special_needs',
        'debtor', 'tuition_fees_up_to_date', 'gender', 'scholarship_holder',
        'age_at_enrollment', 'international',
        
        'curricular_units_1st_sem_credited', 'curricular_units_1st_sem_enrolled', 
        'curricular_units_1st_sem_evaluations', 'curricular_units_1st_sem_approved', 
        'curricular_units_1st_sem_grade', 'curricular_units_1st_sem_without_evaluations',
        
        'curricular_units_2nd_sem_credited', 'curricular_units_2nd_sem_enrolled', 
        'curricular_units_2nd_sem_evaluations', 'curricular_units_2nd_sem_approved', 
        'curricular_units_2nd_sem_grade', 'curricular_units_2nd_sem_without_evaluations',
        
        'unemployment_rate', 'inflation_rate', 'gdp'
    }
    
    @staticmethod
    def parse_file(file: FileStorage) -> tuple[list[dict], str]:
        """
        Parse uploaded file and extract student data into a flat dictionary structure.
        """
        try:
            filename = file.filename.lower()
            file_ext = None
            for ext in FileParser.SUPPORTED_FORMATS:
                if filename.endswith(ext):
                    file_ext = ext
                    break
            
            if not file_ext:
                raise ValueError(f"Unsupported file format. Supported formats: {', '.join(FileParser.SUPPORTED_FORMATS)}")
            
            file_content = file.stream.read()
            
            if file_ext == '.csv':
                df = pd.read_csv(BytesIO(file_content))
            else:
                df = pd.read_excel(BytesIO(file_content), engine='openpyxl' if file_ext == '.xlsx' else 'xlrd')
            
            # Normalize column names to lowercase and replace spaces/special chars with underscores
            df.columns = df.columns.str.lower().str.replace('[^a-z0-9_]+', '_', regex=True)

            # Validate required columns exist
            df_columns = set(df.columns)
            missing_columns = FileParser.REQUIRED_FIELDS - df_columns
            
            if missing_columns:
                raise ValueError(f"Missing required columns (normalized to lowercase/underscores): {', '.join(sorted(missing_columns))}")
            
            records = []
            
            for idx, row in df.iterrows():
                record = {}
                is_valid_row = True
                
                # Extract and clean all fields into a FLAT dictionary
                for col_name in FileParser.REQUIRED_FIELDS:
                    value = row.get(col_name) # Use .get() in case the original column was missing (already checked above, but safe)
                    
                    if pd.isna(value) or value is None:
                        # Allow missing data, but log a warning if a critical feature is missing
                        if col_name in ['name', 'study_year']:
                            logger.error(f"Row {idx + 1} skipped: Missing critical field '{col_name}'.")
                            is_valid_row = False
                            break
                        record[col_name] = None # Store None for nullable DB columns
                        continue

                    # Clean value
                    if isinstance(value, str):
                        value = value.strip()
                    
                    # Try to convert to float/int based on general data expectation
                    try:
                        # Attempt to convert all numeric fields to float for consistency
                        if col_name not in ['name', 'major', 'study_year']:
                            record[col_name] = float(value)
                        else:
                            record[col_name] = str(value)
                    except (ValueError, TypeError):
                        record[col_name] = str(value)
                
                if is_valid_row:
                    records.append(record)
            
            logger.info(f"Successfully parsed {len(records)} valid records from {filename}")
            return records, None
            
        except Exception as e:
            error_msg = f"Error parsing file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return [], error_msg


class DataValidator:
    """
    Utility class for validating student data against expected types and ranges.
    """
    
    # Data Validator logic depends on the specific checks you need, but since all 
    # data is now numeric/string in the flat records, the initial checks are handled 
    # by FileParser.
    
    # ... (Rest of DataValidator remains the same or needs customization) ...
    pass


class DuplicateChecker:
    """
    Utility class for checking and filtering duplicate student records.
    """
    @staticmethod
    def filter_duplicates(new_records: list[dict], existing_records: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Filters out new records that are duplicates of existing records based on
        Name and Study Year.
        """
        # Create a set of existing (name, study_year) tuples for fast lookup
        existing_keys = set()
        for rec in existing_records:
            name = rec.get('name')
            study_year = rec.get('study_year')
            if name and study_year:
                existing_keys.add((name.lower(), str(study_year).lower()))
        
        unique_new_records = []
        duplicate_records = []
        
        for record in new_records:
            name = record.get('name')
            study_year = record.get('study_year')
            
            if name and study_year and (name.lower(), str(study_year).lower()) in existing_keys:
                duplicate_records.append(record)
            else:
                unique_new_records.append(record)
                
        return unique_new_records, duplicate_records


def format_error_response(message: str, detail: str = None) -> dict:
    """Formats an error response dictionary."""
    response = {"status": "error", "message": message}
    if detail:
        response["detail"] = detail
    return response

def format_success_response(data: Any, message: str = "Operation successful") -> dict:
    """Formats a success response dictionary."""
    return {"status": "success", "message": message, "data": data}