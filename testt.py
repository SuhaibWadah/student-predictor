import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Features for prediction
features = {
    'marital_status': 2,
    'application_mode': 1,
    'application_order': 1,
    'course': 171,
    'daytime_evening_attendance': 1,
    'previous_qualification': 12,
    'previous_qualification_grade': 178,
    'nationality': 11,
    'mothers_qualification': 26,
    'fathers_qualification': 35,
    'mothers_occupation': 1,
    'fathers_occupation': 1,
    'admission_grade': 178,
    'displaced': 1,
    'educational_special_needs': 0,
    'debtor': 1,
    'tuition_fees_up_to_date': 0,
    'gender': 0,
    'scholarship_holder': 1,
    'age_at_enrollment': 23,
    'international': 1,
    'curricular_units_1st_sem_credited': 0,
    'curricular_units_1st_sem_enrolled': 7,
    'curricular_units_1st_sem_evaluations': 7,
    'curricular_units_1st_sem_approved': 7,
    'curricular_units_1st_sem_grade': 4,
    'curricular_units_1st_sem_without_evaluations': 0,
    'curricular_units_2nd_sem_credited': 0,
    'curricular_units_2nd_sem_enrolled': 7,
    'curricular_units_2nd_sem_evaluations': 7,
    'curricular_units_2nd_sem_approved': 7,
    'curricular_units_2nd_sem_grade': 4,
    'curricular_units_2nd_sem_without_evaluations': 0,
    'unemployment_rate': 17.1,
    'inflation_rate': 25.8,
    'gdp': 17.4
}

# Space Gradio API URL with hardcoded session hash
HF_SPACE_API_URL = "https://suhaibw-student-performance-predictor.hf.space/gradio_api/queue/join?__theme=system"

# Replace this with the session_hash you see in the browser network call
session_hash = "YOUR_SESSION_HASH_HERE"

payload = {
    "data": [features],
    "fn_index": 0,
    "session_hash": session_hash
}

# If your space is public, no headers are needed. Otherwise, use your HF token
HF_API_KEY = None  # or os.getenv("HUGGINGFACE_API_KEY")
headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

try:
    logger.info("Sending request to Hugging Face Space API...")
    response = requests.post(HF_SPACE_API_URL, json=payload, headers=headers, timeout=30)
    logger.info(f"Status code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        
        predicted_score = result['data'][0] if 'data' in result else None
        logger.info(f"Prediction result: {predicted_score}")
    else:
        logger.error(f"Error from API: {response.text}")

except requests.exceptions.RequestException as e:
    logger.error(f"Request failed: {e}")
