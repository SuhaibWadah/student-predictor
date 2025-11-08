"""
Utility functions for Student Performance Predictor
Provides file parsing, data validation, and helper functions.
"""

import json
import pandas as pd
import numpy as np # Added for robust type handling
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
        'nationality', 
        'mothers_qualification', 'fathers_qualification', 'mothers_occupation', 
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
    def parse_file(file: FileStorage) -> tuple[list[dict], str | None]:
        """
        Parse uploaded file and extract student data into a FLAT dictionary structure.
        """
        df = None # Initialize df outside try/except scope
        try:
            # Determine file extension
            filename = file.filename
            file_content = file.stream.read() # Read content once
            ext = filename[filename.rfind('.'):].lower()

            if ext not in FileParser.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported file format: {ext}")

            # Read file into pandas DataFrame
            if ext == '.csv':
                # Use BytesIO and skip file=file for direct content access
                df = pd.read_csv(BytesIO(file_content))
            elif ext in ['.xls', '.xlsx']:
                # Ensure the engine is specified
                df = pd.read_excel(BytesIO(file_content), engine='openpyxl')
            
            # Normalize column names to lowercase and replace spaces/special chars with underscores
            # This line is now safe because df is defined above.
            df.columns = df.columns.str.lower().str.replace('[^a-z0-9_]+', '_', regex=True)

            # Validate required columns exist
            df_columns = set(df.columns)
            missing_columns = FileParser.REQUIRED_FIELDS - df_columns
            
            if missing_columns:
                raise ValueError(f"Missing required columns (normalized to lowercase/underscores): {', '.join(sorted(missing_columns))}")
            
            records = []
            
            for idx, row in df.iterrows():
                # Initialize the record as a FLAT dictionary
                record = {} 
                is_valid_row = True
                
                # Extract and clean all fields into a FLAT dictionary
                for col_name in FileParser.REQUIRED_FIELDS:
                    value = row.get(col_name)
                    
                    if pd.isna(value) or value is None:
                        if col_name in ['name', 'study_year', 'major']: 
                            logger.error(f"Row {idx + 1} skipped: Missing critical field '{col_name}'.")
                            is_valid_row = False
                            break
                        
                        # Store None for non-critical missing features
                        record[col_name] = None
                        continue

                    # Clean value (strip whitespace for strings)
                    if isinstance(value, str):
                        value = value.strip() 

                    # Store everything directly in the record dictionary (FLAT structure)
                    if col_name in ['name', 'major']:
                        record[col_name] = str(value)
                    else:
                        # Attempt to convert all numerical fields to float/int
                        try:
                            # Use int for most categorical features
                            if col_name in ['study_year'] or col_name.endswith('_status') or col_name.endswith('_mode'):
                                record[col_name] = int(value)
                            # Use float for all other numeric/grade fields
                            else:
                                record[col_name] = float(value)
                        except (ValueError, TypeError):
                            # Store the original value/string; prepare_hf_features will handle the error
                            record[col_name] = value 
                
                if is_valid_row:
                    # Rename 'study_year' to 'year' for consistency with the Student model
                    if 'study_year' in record:
                        record['year'] = record.pop('study_year')
                    else:
                        # Should have been caught by critical check, but skip if somehow missing
                        continue
                        
                    record['semester'] = 1 
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
    pass


class DuplicateChecker:
    """
    Utility class for checking and filtering duplicate student records.
    """
    @staticmethod
    def filter_duplicates(new_records: list[dict], existing_records: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Filters out new records that are duplicates of existing records based on
        Name and Study Year/Year.
        """
        # Create a set of existing (name, year) tuples for fast lookup
        existing_keys = set()
        for rec in existing_records:
            # Use 'year' instead of 'study_year' to match the updated record structure
            name = rec.get('name')
            year = rec.get('year') 
            if name and year:
                existing_keys.add((name.lower(), str(year).lower()))
        
        unique_new_records = []
        duplicate_records = []
        
        for record in new_records:
            # Use 'year' instead of 'study_year'
            name = record.get('name')
            year = record.get('year') 
            
            if name and year and (name.lower(), str(year).lower()) in existing_keys:
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