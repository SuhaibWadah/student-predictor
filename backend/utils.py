"""
Utility functions for Student Performance Predictor
Provides file parsing, data validation, and helper functions.
"""

import json
import pandas as pd
import logging
from io import BytesIO
from werkzeug.datastructures import FileStorage

logger = logging.getLogger(__name__)

class FileParser:
    """
    Utility class for parsing various file formats (CSV, Excel).
    Supports batch student data processing.
    """
    
    # Supported file extensions
    SUPPORTED_FORMATS = {'.csv', '.xlsx', '.xls'}
    
    # Required columns for batch processing
    REQUIRED_COLUMNS = {'name', 'year', 'semester'}
    
    @staticmethod
    def parse_file(file: FileStorage) -> tuple[list[dict], str]:
        """
        Parse uploaded file and extract student data.
        
        Args:
            file: FileStorage object from Flask request
            
        Returns:
            tuple: (list of student records, error message if any)
            
        Raises:
            ValueError: If file format is not supported or required columns are missing
        """
        try:
            # Validate file extension
            filename = file.filename.lower()
            file_ext = None
            for ext in FileParser.SUPPORTED_FORMATS:
                if filename.endswith(ext):
                    file_ext = ext
                    break
            
            if not file_ext:
                raise ValueError(f"Unsupported file format. Supported formats: {', '.join(FileParser.SUPPORTED_FORMATS)}")
            
            # Read file based on extension
            if file_ext == '.csv':
                df = pd.read_csv(file.stream)
            else:  # .xlsx or .xls
                df = pd.read_excel(file.stream, engine='openpyxl' if file_ext == '.xlsx' else 'xlrd')
            
            # Validate required columns exist
            df_columns = set(col.lower() for col in df.columns)
            missing_columns = FileParser.REQUIRED_COLUMNS - df_columns
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Normalize column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Convert dataframe to list of dictionaries
            records = []
            for idx, row in df.iterrows():
                record = {
                    'name': str(row['name']).strip(),
                    'year': int(row['year']),
                    'semester': int(row['semester']),
                    'features': {}
                }
                
                # Extract additional features (all columns except required ones)
                for col in df.columns:
                    if col not in FileParser.REQUIRED_COLUMNS:
                        # Try to convert to numeric if possible
                        try:
                            record['features'][col] = float(row[col])
                        except (ValueError, TypeError):
                            record['features'][col] = str(row[col])
                
                records.append(record)
            
            logger.info(f"Successfully parsed {len(records)} records from {filename}")
            return records, None
            
        except Exception as e:
            error_msg = f"Error parsing file: {str(e)}"
            logger.error(error_msg)
            return [], error_msg


class DataValidator:

    REQUIRED_FIELDS = [
        'marital_status', 'application_mode', 'course',
        'previous_qualification_grade', 'mothers_qualification', 'fathers_qualification',
        'mothers_occupation', 'fathers_occupation', 'displaced', 'educational_special_needs',
        'debtor', 'tuition_fees_up_to_date', 'gender', 'scholarship_holder',
        'age_at_enrollment', 'international',
        'curricular_units_1st_sem_enrolled', 'curricular_units_1st_sem_approved', 'curricular_units_1st_sem_grade',
        'curricular_units_2nd_sem_enrolled', 'curricular_units_2nd_sem_approved', 'curricular_units_2nd_sem_grade'
    ]

    @staticmethod
    def validate_student_data(data):
        missing_fields = [f for f in DataValidator.REQUIRED_FIELDS if f not in data]
        if missing_fields:
            return False, f"Missing fields: {', '.join(missing_fields)}"

        # Optional: validate numeric ranges
        try:
            if not (0 <= data['previous_qualification_grade'] <= 200):
                return False, "Previous qualification grade must be 0-200"

            if not (0 <= data['curricular_units_1st_sem_grade'] <= 4):
                return False, "Semester 1 GPA must be 0-4"

            if not (0 <= data['curricular_units_2nd_sem_grade'] <= 4):
                return False, "Semester 2 GPA must be 0-4"
        except Exception as e:
            return False, str(e)

        return True, None


class DuplicateChecker:
    """
    Utility class for checking and handling duplicate student records.
    """
    
    @staticmethod
    def is_duplicate(name: str, year: int, semester: int, existing_records: list) -> bool:
        """
        Check if a student record already exists in the database.
        
        Args:
            name: Student name
            year: Academic year
            semester: Semester number
            existing_records: List of existing student records
            
        Returns:
            bool: True if duplicate found, False otherwise
        """
        for record in existing_records:
            if (record['name'].lower() == name.lower() and 
                record['year'] == year and 
                record['semester'] == semester):
                return True
        return False
    
    @staticmethod
    def filter_duplicates(new_records: list, existing_records: list) -> tuple[list, list]:
        """
        Filter out duplicate records from new batch.
        
        Args:
            new_records: List of new student records
            existing_records: List of existing student records
            
        Returns:
            tuple: (unique_records, duplicate_records)
        """
        unique_records = []
        duplicate_records = []
        
        for record in new_records:
            if DuplicateChecker.is_duplicate(
                record['name'],
                record['year'],
                record['semester'],
                existing_records + unique_records
            ):
                duplicate_records.append(record)
            else:
                unique_records.append(record)
        
        return unique_records, duplicate_records


def format_error_response(message: str, details: str = None) -> dict:
    """
    Format error response for API endpoints.
    
    Args:
        message: Main error message
        details: Additional error details
        
    Returns:
        dict: Formatted error response
    """
    response = {'error': message}
    if details:
        response['details'] = details
    return response


def format_success_response(data: any, message: str = None) -> dict:
    """
    Format success response for API endpoints.
    
    Args:
        data: Response data
        message: Optional success message
        
    Returns:
        dict: Formatted success response
    """
    response = {'data': data}
    if message:
        response['message'] = message
    return response
