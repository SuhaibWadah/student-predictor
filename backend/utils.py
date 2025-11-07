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
                # Use a BytesIO object with the stream to allow pandas to seek/read
                df = pd.read_csv(BytesIO(file.stream.read()))
            else:  # .xlsx or .xls
                # Use a BytesIO object with the stream to allow pandas to seek/read
                df = pd.read_excel(BytesIO(file.stream.read()), engine='openpyxl' if file_ext == '.xlsx' else 'xlrd')
            
            # Validate required columns exist
            df_columns = set(col.lower() for col in df.columns)
            
            # FIX: Both operands are now sets, so the set difference operation works correctly.
            missing_columns = FileParser.REQUIRED_FIELDS - df_columns
            
            if missing_columns:
                # Use sorted() for consistent error messaging
                raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")
            
            # Normalize column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Convert dataframe to list of dictionaries
            records = []
            # Only iterate over the columns that are explicitly required
            required_cols_list = list(FileParser.REQUIRED_FIELDS) 
            
            for idx, row in df.iterrows():
                # NOTE: Only include required columns for basic record structure
                record = {
                    'name': str(row['name']).strip(),
                    'year': int(row['year']),
                    'semester': int(row['semester']),
                    'features': {}
                }
                
                # Extract additional features (all columns that are NOT in REQUIRED_FIELDS)
                for col in df.columns:
                    if col not in FileParser.REQUIRED_FIELDS:
                        # Try to convert to numeric if possible
                        try:
                            record['features'][col] = float(row[col])
                        except (ValueError, TypeError):
                            record['features'][col] = str(row[col])
                    # Ensure required columns are included as features for validation later
                    elif col not in ['name', 'year', 'semester']:
                        # NOTE: Ensure these required fields are treated correctly (e.g., as integers/floats if expected)
                        # For simplicity, we are placing them directly into 'features' for DataValidator
                         record['features'][col] = row[col]
                        
                records.append(record)
            
            logger.info(f"Successfully parsed {len(records)} records from {filename}")
            return records, None
            
        except Exception as e:
            error_msg = f"Error parsing file: {str(e)}"
            logger.error(error_msg)
            return [], error_msg


class DataValidator:

    # Defined as a SET for efficient checking
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
            return False, f"Missing fields: {', '.join(sorted(missing_fields))}"

        # Optional: validate numeric ranges
        try:
            # Convert values to expected types before validation, as they may come from a JSON/API payload
            # or the 'features' dict created above.
            
            # Using data.get() with a default value of 0 in case the key is missing (though checked above)
            grade = float(data['previous_qualification_grade'])
            sem1_gpa = float(data['curricular_units_1st_sem_grade'])
            sem2_gpa = float(data['curricular_units_2nd_sem_grade'])
            
            if not (0 <= grade <= 200):
                return False, "Previous qualification grade must be 0-200"

            # Assuming 4.0 is the max GPA, update if a different scale is used
            if not (0 <= sem1_gpa <= 4.0):
                return False, "Semester 1 GPA must be 0-4"

            if not (0 <= sem2_gpa <= 4.0):
                return False, "Semester 2 GPA must be 0-4"
        except (ValueError, TypeError) as e:
            return False, f"Data type error during validation: {str(e)}. Check if all required fields are numeric."
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
        # Lowercase and strip name outside the loop for efficiency
        norm_name = name.lower().strip()
        
        for record in existing_records:
            # Ensure safe access and comparison
            if (record.get('name', '').lower().strip() == norm_name and 
                record.get('year') == year and 
                record.get('semester') == semester):
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
        
        # NOTE: This approach is O(N*M) where N is new and M is existing.
        # For large datasets, a set of keys (name, year, semester) should be pre-built from existing_records
        
        # Build a set of existing keys for O(1) lookups
        existing_keys = set()
        for record in existing_records:
            try:
                key = (record['name'].lower().strip(), record['year'], record['semester'])
                existing_keys.add(key)
            except (KeyError, AttributeError, TypeError):
                # Log an error if an existing record is malformed
                logger.warning(f"Malformed existing record skipped in duplicate check: {record}")
                
        
        for record in new_records:
            try:
                record_key = (record['name'].lower().strip(), record['year'], record['semester'])
                
                # Check against existing records and newly added unique records
                if record_key in existing_keys:
                    duplicate_records.append(record)
                else:
                    unique_records.append(record)
                    # Add to existing_keys to prevent duplicates within the new batch itself
                    existing_keys.add(record_key) 
            except (KeyError, AttributeError, TypeError):
                 logger.error(f"Malformed new record skipped: {record}")
                 
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