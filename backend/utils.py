"""
Utility functions for Student Performance Predictor
Provides file parsing, data validation, and helper functions.
"""

import json
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
    
    # Supported file extensions
    SUPPORTED_FORMATS = {'.csv', '.xlsx', '.xls'}
    
    # 🌟 FIX: Full 36-feature set (Including macro-economic and detailed semester units) 🌟
    REQUIRED_FIELDS = {
        'name', 'year', 'semester', 'marital_status', 'application_mode', 'application_order', 
        'course', 'daytime_evening_attendance', 'previous_qualification', 'previous_qualification_grade', 
        'nationality', 'mothers_qualification', 'fathers_qualification', 'mothers_occupation', 
        'fathers_occupation', 'admission_grade', 'displaced', 'educational_special_needs',
        'debtor', 'tuition_fees_up_to_date', 'gender', 'scholarship_holder',
        'age_at_enrollment', 'international',
        
        # 1st Semester
        'curricular_units_1st_sem_credited', 'curricular_units_1st_sem_enrolled', 
        'curricular_units_1st_sem_evaluations', 'curricular_units_1st_sem_approved', 
        'curricular_units_1st_sem_grade', 'curricular_units_1st_sem_without_evaluations',
        
        # 2nd Semester
        'curricular_units_2nd_sem_credited', 'curricular_units_2nd_sem_enrolled', 
        'curricular_units_2nd_sem_evaluations', 'curricular_units_2nd_sem_approved', 
        'curricular_units_2nd_sem_grade', 'curricular_units_2nd_sem_without_evaluations',
        
        # Macro-economic factors
        'unemployment_rate', 'inflation_rate', 'gdp'
    }
    
    @staticmethod
    def parse_file(file: FileStorage) -> tuple[list[dict], str]:
        """
        Parse uploaded file and extract student data.
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
            file_content = file.stream.read()
            
            if file_ext == '.csv':
                df = pd.read_csv(BytesIO(file_content))
            else:  # .xlsx or .xls
                df = pd.read_excel(BytesIO(file_content), engine='openpyxl' if file_ext == '.xlsx' else 'xlrd')
            
            # Normalize column names to lowercase
            df.columns = df.columns.str.lower()

            # Validate required columns exist
            df_columns = set(df.columns)
            missing_columns = FileParser.REQUIRED_FIELDS - df_columns
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")
            
            # Convert dataframe to list of dictionaries
            records = []
            
            for idx, row in df.iterrows():
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
                
                # Extract and clean features
                for col in df.columns:
                    # Ignore 'name', 'year', 'semester' for the main 'features' block in the record
                    if col not in ['name', 'year', 'semester']:
                        value = row[col]
                        
                        # Ensure string values are stripped if not NaN (important for names/codes)
                        if isinstance(value, str):
                            value = value.strip()
                        
                        # Try to convert to numeric if possible
                        try:
                            # 🌟 FIX: Force numeric conversion for known numeric fields (like GDP) 🌟
                            if col in FileParser.REQUIRED_FIELDS:
                                record['features'][col] = float(value)
                            else:
                                record['features'][col] = float(value) # Default for other known columns
                        except (ValueError, TypeError):
                            # Ensure all non-numeric values are stripped strings
                            record['features'][col] = str(value) if pd.notna(value) else None
                
                records.append(record)
            
            logger.info(f"Successfully parsed {len(records)} records from {filename}")
            return records, None
            
        except Exception as e:
            error_msg = f"Error parsing file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return [], error_msg


class DataValidator:
    """
    Utility class for validating student data against expected types and ranges.
    """
    
    # 🌟 FIX: Use the full set of REQUIRED_FIELDS from FileParser 🌟
    # We remove the identification fields since they are validated elsewhere.
    REQUIRED_FIELDS = FileParser.REQUIRED_FIELDS - {'name', 'year', 'semester'}

    @staticmethod
    def validate_student_data(data):
        # Use set difference for missing fields check
        missing_fields = DataValidator.REQUIRED_FIELDS - set(data.keys())
        if missing_fields:
            logger.warning(f"DataValidator: Failing validation due to missing fields: {missing_fields}")
            return False, f"Missing fields: {', '.join(sorted(missing_fields))}"

        # Optional: validate numeric ranges
        try:
            # Attempt to safely convert and validate critical numeric fields
            grade = float(data.get('previous_qualification_grade', 0))
            sem1_gpa = float(data.get('curricular_units_1st_sem_grade', 0))
            sem2_gpa = float(data.get('curricular_units_2nd_sem_grade', 0))
            
            if not (0 <= grade <= 200):
                return False, "Previous qualification grade must be 0-200"

            # Assuming GPA is 0-4 scale or similar. Check your model documentation.
            if not (0 <= sem1_gpa): 
                return False, "Semester 1 GPA must be non-negative"

            if not (0 <= sem2_gpa):
                return False, "Semester 2 GPA must be non-negative"
                
        except (ValueError, TypeError) as e:
            error_msg = f"DataValidator Error: Type or range validation failed. Check numeric fields. {str(e)}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            return False, str(e)

        logger.debug("DataValidator: Validation successful.")
        return True, None


class DuplicateChecker:
    """
    Utility class for filtering duplicate records based on name, year, and semester.
    """
    
    @staticmethod
    def filter_duplicates(new_records: list, existing_records: list) -> tuple[list, list]:
        """
        Filter out duplicate records from new batch.
        """
        unique_records = []
        duplicate_records = []
        
        # Build a set of existing keys for O(1) lookups
        existing_keys = set()
        for record in existing_records:
            try:
                name = record.get('name')
                year = record.get('year')
                semester = record.get('semester')
                
                if name is not None and year is not None and semester is not None:
                    key = (str(name).lower().strip(), year, semester)
                    existing_keys.add(key)
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
    """Format a standard error response dictionary."""
    response = {'error': message}
    if details:
        response['details'] = details
    return response


def format_success_response(data: any, message: str = None) -> dict:
    """Format a standard success response dictionary."""
    response = {'data': data}
    if message:
        response['message'] = message
    return response