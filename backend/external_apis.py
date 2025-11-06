import os
import requests
import logging
import json 
import time
from typing import Any, Optional, Tuple
from gradio_client import Client

logger = logging.getLogger(__name__)

class HuggingFaceAPI:
    """Interface for Hugging Face hosted model predictions using the official Gradio Client."""

    HF_SPACE_ID = "suhaibW/student-performance-predictor"
    API_NAME = "/predict" 

    def __init__(self):
        self.api_key = os.getenv('HUGGINGFACE_API_KEY') or os.getenv('HF_TOKEN')
        
        try:
            # The client connects to the space ID and handles authentication
            self.client = Client(self.HF_SPACE_ID, hf_token=self.api_key)
            logger.info(f"Gradio Client initialized for Space: {self.HF_SPACE_ID}")
        except Exception as e:
            logger.error(f"Failed to initialize Gradio Client: {str(e)}")
            self.client = None

    def predict(self, features: dict[str, Any], timeout: int = 30) -> Tuple[Optional[Any], Optional[str]]:
        if not self.client:
            return None, "Gradio Client failed to initialize."
        
        # CRITICAL: This line assumes 'features' is ALREADY ordered by the caller (routes.py).
        data_for_api = list(features.values())
        
        if len(data_for_api) != 36:
            return None, f"Input error: Expected 36 feature values, received {len(data_for_api)}."
        
        try:
            start_time = time.time()
            
            # Unpack the 36 parameters as positional arguments
            result = self.client.predict(
                *data_for_api, 
                api_name=self.API_NAME,
                # request_options={"timeout": timeout} # Removed as per previous fix
            )
            response_time = int((time.time() - start_time) * 1000)

            logger.info("Prediction successful: %s, %dms", result, response_time)
            # The result here is the categorical string (e.g., 'Success')
            return result, None

        except Exception as e:
            return None, f"Gradio Client prediction failed: {str(e)}. (Ensure the Space is running!)"

class OpenRouterAPI:
    """
    Interface for OpenRouter.ai LLM API.
    Generates improvement plans based on predictions.
    """
    key = "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"
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self):
        self.api_key = self.key
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not set. Plan generation will fail.")

        self.headers = {
            'Authorization': f'Bearer {self.api_key}', 
            'Content-Type': 'application/json',
            'HTTP-Referer': os.getenv('APP_URL', 'http://localhost:5000'),
            'X-Title': 'Student Performance Predictor'
        }

    def generate_improvement_plan(
        self,
        student_name: str,
        year: int,
        semester: int,
        features: dict[str, Any],
        predicted_score: Any,
        timeout: int = 30
    ) -> tuple[str | None, str | None]:
        """Generate a personalized improvement plan using OpenRouter LLM"""
        if not self.api_key:
            return None, "OpenRouter API key not configured"

        prompt = self._prepare_plan_prompt(student_name, year, semester, features, predicted_score)

        payload = {
            'model': 'anthropic/claude-sonnet-4.5',
            'messages': [
                {'role': 'system', 'content': 'You are an educational advisor helping students improve performance. Provide actionable improvement plans.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.7,
            'max_tokens': 885 # 🌟 FIX 1: INCREASED FROM 500 TO 2000 (Prevents Truncation) 🌟
        }

        try:
            start_time = time.time()
            logger.debug("Sending OpenRouter request payload: %s", payload)

            response = requests.post(
                self.OPENROUTER_API_URL, json=payload, headers=self.headers, timeout=timeout
            )
            response_time = int((time.time() - start_time) * 1000)

            if response.status_code != 200:
                error_msg = f"OpenRouter API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                return None, error_msg

            result = response.json()
            if 'choices' not in result or not result['choices']:
                logger.error("Invalid response from OpenRouter API: %s", json.dumps(result))
                return None, "Invalid response format from OpenRouter API"

            improvement_plan = result['choices'][0]['message']['content']
            logger.info("Plan generated successfully: %dms", response_time)
            
            return improvement_plan, None

        except requests.exceptions.Timeout:
            return None, "OpenRouter API request timed out"
        except requests.exceptions.RequestException as e:
            return None, f"OpenRouter API request failed: {str(e)}"
        except Exception as e:
            return None, f"Unexpected error during OpenRouter plan generation: {str(e)}"

    @staticmethod
    def _prepare_plan_prompt(
        student_name: str,
        year: int,
        semester: int,
        features: dict[str, Any],
        predicted_score: Any
    ) -> str:
        """Prepares the detailed prompt for the LLM."""
        
        # 🌟 FIX 2: Handle categorical score as a safe string 🌟
        score_display = str(predicted_score)
        
        features_text = "\n".join([f"- {k}: {v}" for k, v in features.items()])
        return f"""
Generate a personalized improvement plan for the following student:

**Student:** {student_name}
**Year:** {year}
**Semester:** {semester}
**Predicted Performance:** {score_display}

**Relevant Features:**
{features_text}

---
**STRICT INSTRUCTIONS:**
Provide the plan using **Markdown headings (##)** for each of the five requested sections below. Be extremely concise. The entire output must be less or close to than 880 tokens.

## 1. Analysis of Current Performance (Max 3 concise sentences)
## 2. Key Areas for Improvement (List 2-3 bullet points)
## 3. Specific Action Items (Provide 3-5 numbered, detailed steps)
## 4. Timeline for Implementation (Suggest specific timeframes: e.g., 'Weeks 1-4')
## 5. Success Metrics (List 2-3 measurable indicators)

---
""".strip()