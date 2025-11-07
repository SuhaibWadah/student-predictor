"""
Utility functions for Student Performance Predictor
Provides file parsing, data validation, and helper functions.
"""

import json
# 🌟 FIX: Ensure pandas is imported as pd 🌟
import pandas as pd
import logging
from io import BytesIO
from werkzeug.datastructures import FileStorage

# Configure logging (good practice)
logger = logging.getLogger(__name__)

class FileParser:
    """
    Utility class for parsing various file formats (CSV, Excel).
    Supports batch student data processing.
    """
    
    # 🌟 FIX: ADDED MISSING ATTRIBUTE (Fixes previous AttributeError) 🌟
    # Supported file extensions
    SUPPORTED_FORMATS = {'.csv', '.xlsx', '.xls'}
    
    # Required columns for batch processing. Defined as a SET for efficient lookups and set operations.
    REQUIRED_FIELDS = {
        'name', 'year', 'semester', 'marital_status', 'application_mode', 'course',
        'previous_qualification_grade', 'mothers_qualification', 'fathers_qualification',
        'mothers_occupation', 'fathers_occupation', 'displaced', 'educational_special_needs',
        'debtor', 'tuition_fees_up_to_date', 'gender', 'scholarship_holder',
        'age_at_enrollment', 'international',
        'curricular_units_1st_sem_enrolled', 'curricular_units_1st_sem_approved', 'curricular_units_1st_sem_grade',
        'curricular_units_2nd_sem_enrolled', 'curricular_units_2nd_sem_approved', 'curricular_units_2nd_sem_grade'
    }
    
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
                df = pd.read_csv(BytesIO(file.stream.read()))
            else:  # .xlsx or .xls
                df = pd.read_excel(BytesIO(file.stream.read()), engine='openpyxl' if file_ext == '.xlsx' else 'xlrd')
            
            # Validate required columns exist
            df_columns = set(col.lower() for col in df.columns)
            missing_columns = FileParser.REQUIRED_FIELDS - df_columns
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")
            
            # Normalize column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Convert dataframe to list of dictionaries
            records = []
            
            for idx, row in df.iterrows():
                # --- LOGGING ADDED: Log the raw row data ---
                logger.debug(f"FileParser: Processing Row {idx + 1}. Raw data: {row.to_dict()}")

                try:
                    record = {
                        'name': str(row['name']).strip(),
                        'year': int(row['year']),
                        'semester': int(row['semester']),
                        'features': {}
                    }
                except KeyError as e:
                    logger.error(f"FileParser Error: Missing required primary key in row {idx + 1}: {e}")
                    raise
                except ValueError as e:
                    logger.error(f"FileParser Error: Type mismatch for primary key (year/semester) in row {idx + 1}: {e}")
                    raise
                
                # Extract features
                for col in df.columns:
                    # Ignore 'name', 'year', 'semester' for the main 'features' block in the record
                    if col not in ['name', 'year', 'semester']:
                        value = row[col]
                        # Try to convert to numeric if possible (this is a common error source)
                        try:
                            record['features'][col] = float(value)
                        except (ValueError, TypeError):
                            # pd.notna is used here
                            record['features'][col] = str(value).strip() if pd.notna(value) else None
                
                records.append(record)
                logger.debug(f"FileParser: Row {idx + 1} converted successfully. Record features keys: {list(record['features'].keys())}")
            
            logger.info(f"Successfully parsed {len(records)} records from {filename}")
            return records, None
            
        except Exception as e:
            error_msg = f"Error parsing file: {str(e)}"
            logger.error(error_msg, exc_info=True) # Log full traceback
            return [], error_msg


class DataValidator:
    """Utility class for validating student feature data."""

    REQUIRED_FIELDS = {
        'marital_status', 'application_mode', 'course',
        'previous_qualification_grade', 'mothers_qualification', 'fathers_qualification',
        'mothers_occupation', 'fathers_occupation', 'displaced', 'educational_special_needs',
        'debtor', 'tuition_fees_up_to_date', 'gender', 'scholarship_holder',
        'age_at_enrollment', 'international',
        'curricular_units_1st_sem_enrolled', 'curricular_units_1st_sem_approved', 'curricular_units_1st_sem_grade',
        'curricular_units_2nd_sem_enrolled', 'curricular_units_2nd_sem_approved', 'curricular_units_2nd_sem_grade'
    }

    @staticmethod
    def validate_student_data(data):
        # Use set difference for missing fields check
        missing_fields = DataValidator.REQUIRED_FIELDS - set(data.keys())
        if missing_fields:
            # --- LOGGING ADDED: Log missing fields before returning failure ---
            logger.warning(f"DataValidator: Failing validation due to missing fields: {missing_fields}")
            return False, f"Missing fields: {', '.join(sorted(missing_fields))}"

        # Optional: validate numeric ranges
        try:
            # Attempt to safely convert and validate critical numeric fields
            grade = float(data['previous_qualification_grade'])
            sem1_gpa = float(data['curricular_units_1st_sem_grade'])
            sem2_gpa = float(data['curricular_units_2nd_sem_grade'])
            
            if not (0 <= grade <= 200):
                return False, "Previous qualification grade must be 0-200"

            if not (0 <= sem1_gpa <= 4.0):
                return False, "Semester 1 GPA must be 0-4"

            if not (0 <= sem2_gpa <= 4.0):
                return False, "Semester 2 GPA must be 0-4"
                
        except (ValueError, TypeError) as e:
            # --- LOGGING ADDED: Log the specific field causing the type error ---
            error_msg = f"DataValidator Error: Type or range validation failed. Check numeric fields. {str(e)}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            return False, str(e)

        logger.debug("DataValidator: Validation successful.")
        return True, None


class DuplicateChecker:
    """Utility class to filter duplicate student records."""
    
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
        
        # Build a set of existing keys for O(1) lookups
        existing_keys = set()
        for record in existing_records:
            try:
                # Ensure fields exist before accessing
                name = record.get('name')
                year = record.get('year')
                semester = record.get('semester')
                
                if name is not None and year is not None and semester is not None:
                    key = (str(name).lower().strip(), year, semester)
                    existing_keys.add(key)
                else:
                    logger.warning(f"Malformed existing record skipped in duplicate check (missing key): {record}")
            except Exception:
                logger.warning(f"Malformed existing record skipped in duplicate check: {record}")
                
        
        for record in new_records:
            try:
                record_key = (record['name'].lower().strip(), record['year'], record['semester'])
                
                # Check against existing records and newly added unique records
                if record_key in existing_keys:
                    duplicate_records.append(record)
                    logger.debug(f"DuplicateChecker: Found duplicate record: {record_key}")
                else:
                    unique_records.append(record)
                    # Add to existing_keys to prevent duplicates within the new batch itself
                    existing_keys.add(record_key) 
                    logger.debug(f"DuplicateChecker: Added unique record: {record_key}")
            except (KeyError, AttributeError, TypeError):
                 logger.error(f"Malformed new record skipped during key creation: {record}")
                 
        return unique_records, duplicate_records


def format_error_response(message: str, details: str = None) -> dict:
    """Formats a consistent error response dictionary."""
    response = {'error': message}
    if details:
        response['details'] = details
    return response


def format_success_response(data: any, message: str = None) -> dict:
    """Formats a consistent success response dictionary."""
    response = {'data': data}
    if message:
        response['message'] = message
    return response