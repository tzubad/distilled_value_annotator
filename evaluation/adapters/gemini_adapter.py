# Gemini LLM adapter for video annotation evaluation

import csv
import json
import logging
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

from evaluation.adapters.base import ModelAdapter
from evaluation.adapters.script_loader import ScriptLoader
from evaluation.models import VideoAnnotation, PredictionResult, PredictionSet
from evaluation.video_id_utils import normalize_video_id


class GeminiAdapter(ModelAdapter):
    """
    Adapter for Google Gemini LLM models.
    
    Loads system instructions from prompts/script_to_annotation_instructions.txt
    and uses Vertex AI to generate annotations for video scripts.
    
    Requirements: 1.1, 1.2, 7.2, 7.3
    """
    
    # Mapping from prompt format to standard category names
    CATEGORY_MAPPING = {
        "1_Value1_Self_Direction_Thought_values": "Self_Direction_Thought",
        "1_Value1_Self_Direction_Action_values": "Self_Direction_Action",
        "1_Value1_Stimulation_values": "Stimulation",
        "1_Value1_Hedonism_values": "Hedonism",
        "1_Value1_Achievement_values": "Achievement",
        "1_Value1_Power_Resources_values": "Power_Resources",
        "1_Value1_Power_dominance_values": "Power_Dominance",
        "1_Value1_Face_values": "Face",
        "1_Value1_Security_Personal_values": "Security_Personal",
        "1_Value1_Security_Social_values": "Security_Social",
        "1_Value1_Conformity_Rules_values": "Conformity_Rules",
        "1_Value1_Conformity_Interpersonal_values": "Conformity_Interpersonal",
        "1_Value1_Tradition_values": "Tradition",
        "1_Value1_Humility_values": "Humility",
        "1_Value1_Benevolence_Dependability_values": "Benevolence_Dependability",
        "1_Value1_Benevolence_Care_values": "Benevolence_Care",
        "1_Value1_Universalism_Concern_values": "Universalism_Concern",
        "1_Value1_Universalism_Nature_values": "Universalism_Nature",
        "1_Value1_Universalism_Tolerance_values": "Universalism_Tolerance",
    }
    
    # Mapping from LLM output values to standard annotation values
    VALUE_MAPPING = {
        "present": 1,
        "dominant": 2,
        "conflict": -1,
        None: 0,
        "None": 0,
        "none": 0,  # lowercase version
        "": 0,
        # Numeric string values (sometimes in predictions CSV)
        "0": 0,
        "1": 1,
        "2": 2,
        "-1": -1,
        "{'present': 1}": 1,
    }
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initialize the Gemini adapter.
        
        Args:
            model_name: Gemini model identifier (e.g., 'gemini-1.5-pro-002')
            config: Configuration including:
                - max_retries: Maximum number of retry attempts (default: 3)
                - retry_delay: Initial delay between retries in seconds (default: 30)
                - safety_settings: Dictionary of safety setting overrides
                - system_instructions_path: Path to system instructions file
        """
        super().__init__(model_name, config)
        
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 30)
        self.safety_settings = config.get('safety_settings', {})
        self.system_instructions_path = config.get(
            'system_instructions_path',
            'prompts/script_to_annotation_instructions.txt'
        )
        
        self.script_loader = ScriptLoader()
        self.model = None
        self.system_instructions = None
    
    def initialize(self) -> bool:
        """
        Load model and prepare for inference.
        
        Loads system instructions and initializes Vertex AI Gemini model.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Load system instructions
            self.system_instructions = self._load_system_instructions()
            if self.system_instructions is None:
                self.logger.error("Failed to load system instructions")
                return False
            
            # Initialize Vertex AI
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel, SafetySetting, HarmCategory, HarmBlockThreshold
                import os
                
                # Get project ID from environment
                project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
                if not project_id:
                    self.logger.error("GOOGLE_CLOUD_PROJECT environment variable not set")
                    return False
                
                # Initialize Vertex AI with explicit project
                vertexai.init(project=project_id, location="us-central1")
                self.logger.info(f"Initialized Vertex AI with project: {project_id}")
                
                # Configure safety settings
                safety_settings = self._configure_safety_settings()
                
                # Initialize the model
                self.model = GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=self.system_instructions
                )
                
                self.logger.info(f"Successfully initialized Gemini model: {self.model_name}")
                return True
            
            except ImportError as e:
                self.logger.error(f"Failed to import Vertex AI libraries: {str(e)}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error initializing Gemini adapter: {str(e)}", exc_info=True)
            return False
    
    def predict(self, video: VideoAnnotation) -> Optional[PredictionResult]:
        """
        Generate predictions for a single video.
        
        Loads the video script, sends it to Gemini, and parses the response
        to extract annotations for all 19 categories.
        
        Args:
            video: VideoAnnotation object containing video metadata and script
        
        Returns:
            PredictionResult with predictions for all 19 categories,
            or None if prediction fails
        """
        start_time = time.time()
        
        try:
            # Load script content
            script_text = self._load_script(video)
            if script_text is None:
                return PredictionResult(
                    video_id=video.video_id,
                    predictions={},
                    success=False,
                    error_message="Failed to load script",
                    inference_time=time.time() - start_time
                )
            
            # Generate prediction with retry logic
            response_text = self._retry_with_backoff(
                lambda: self._generate_with_model(script_text),
                video.video_id
            )
            
            if response_text is None:
                return PredictionResult(
                    video_id=video.video_id,
                    predictions={},
                    success=False,
                    error_message="Failed to generate prediction after retries",
                    inference_time=time.time() - start_time
                )
            
            # Parse the LLM response
            predictions = self._parse_llm_response(response_text)
            
            if predictions is None:
                return PredictionResult(
                    video_id=video.video_id,
                    predictions={},
                    success=False,
                    error_message="Failed to parse LLM response",
                    inference_time=time.time() - start_time
                )
            
            # Verify we have all 19 categories
            if len(predictions) != 19:
                return PredictionResult(
                    video_id=video.video_id,
                    predictions=predictions,
                    success=False,
                    error_message=f"Incomplete predictions: got {len(predictions)}/19 categories",
                    inference_time=time.time() - start_time
                )
            
            return PredictionResult(
                video_id=video.video_id,
                predictions=predictions,
                success=True,
                inference_time=time.time() - start_time
            )
        
        except Exception as e:
            self.logger.error(
                f"Error predicting video {video.video_id}: {str(e)}",
                exc_info=True
            )
            return PredictionResult(
                video_id=video.video_id,
                predictions={},
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                inference_time=time.time() - start_time
            )
    
    def get_model_type(self) -> str:
        """Return the type of model."""
        return "LLM"
    
    def get_model_name(self) -> str:
        """Return the specific model identifier."""
        return self.model_name
    
    def _load_system_instructions(self) -> Optional[str]:
        """
        Load system instructions from file.
        
        Returns:
            System instructions as string, or None if loading fails
        """
        try:
            instructions_path = Path(self.system_instructions_path)
            
            if not instructions_path.exists():
                self.logger.error(f"System instructions file not found: {self.system_instructions_path}")
                return None
            
            with open(instructions_path, 'r', encoding='utf-8') as f:
                instructions = f.read()
            
            self.logger.info(f"Loaded system instructions from {self.system_instructions_path}")
            return instructions
        
        except Exception as e:
            self.logger.error(f"Error loading system instructions: {str(e)}")
            return None
    
    def _configure_safety_settings(self):
        """
        Configure safety settings for Gemini model.
        
        Returns:
            List of SafetySetting objects
        """
        from vertexai.generative_models import SafetySetting, HarmCategory, HarmBlockThreshold
        
        # Default safety settings (block none to allow all content for research)
        default_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Apply any overrides from config
        # (config uses string keys, need to map to enum values)
        safety_settings = []
        for category, threshold in default_settings.items():
            safety_settings.append(SafetySetting(category=category, threshold=threshold))
        
        return safety_settings
    
    def _load_script(self, video: VideoAnnotation) -> Optional[str]:
        """
        Load script for a video.
        
        Uses cached script_text if available, otherwise loads from script_uri.
        
        Args:
            video: VideoAnnotation object
        
        Returns:
            Script text, or None if loading fails
        """
        # Use cached script if available
        if video.script_text is not None:
            return video.script_text
        
        # Load from script_uri
        return self.script_loader.load_script(video.script_uri)
    
    def _generate_with_model(self, script_text: str) -> str:
        """
        Generate annotation using Gemini model.
        
        Args:
            script_text: Video script text
        
        Returns:
            Model response text
        
        Raises:
            Exception if generation fails
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        # Configure safety settings
        safety_settings = self._configure_safety_settings()
        
        # Generate content
        response = self.model.generate_content(
            script_text,
            safety_settings=safety_settings
        )
        
        return response.text
    
    def _retry_with_backoff(self, func, video_id: str) -> Optional[str]:
        """
        Execute function with exponential backoff retry logic.
        
        Handles API rate limits and transient errors by retrying with
        exponentially increasing delays.
        
        Args:
            func: Function to execute
            video_id: Video ID for logging
        
        Returns:
            Function result, or None if all retries fail
        
        Requirements: 10.1
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return func()
            
            except Exception as e:
                last_error = e
                
                # Log the error with context
                self.logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed for video {video_id} "
                    f"with model {self.model_name}: {str(e)}"
                )
                
                # If this was the last attempt, don't sleep
                if attempt < self.max_retries - 1:
                    # Calculate exponential backoff delay
                    delay = self.retry_delay * (2 ** attempt)
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
        
        # All retries failed
        self.logger.error(
            f"All {self.max_retries} attempts failed for video {video_id} "
            f"with model {self.model_name}: {str(last_error)}"
        )
        return None
    
    def _parse_llm_value(self, llm_value) -> Optional[int]:
        """
        Parse a value from the LLM response into numeric annotation value.
        
        Handles various formats the LLM might return:
        - Standard strings: "present", "conflict", "dominant", "endorsed", "None", None
        - Integers: 0, 1, 2, -1
        - Dictionary-like strings: "{'present': 1}", "{'conflict': -1}"
        - Dictionaries: {'present': 1}
        
        Args:
            llm_value: Value from LLM response
        
        Returns:
            Numeric value (-1, 0, 1, 2) or None if unable to parse
        """
        # Handle None or "None"
        if llm_value is None or llm_value == "None":
            return 0
        
        # Handle direct integers
        if isinstance(llm_value, int):
            if llm_value in {-1, 0, 1, 2}:
                return llm_value
            return None
        
        # Handle string values
        if isinstance(llm_value, str):
            # Standard text values
            value_lower = llm_value.strip().lower()
            
            # Handle compound values (e.g., "present_dominant", "endorsed_dominant")
            if "dominant" in value_lower:
                return 2
            elif value_lower == "present" or value_lower == "endorsed":
                return 1
            elif value_lower == "conflict":
                return -1
            elif value_lower == "none" or value_lower == "":
                return 0
            
            # Try to parse as integer
            try:
                int_val = int(value_lower)
                if int_val in {-1, 0, 1, 2}:
                    return int_val
            except ValueError:
                pass
        
        # Handle dictionary values
        if isinstance(llm_value, dict):
            # Check for structured format: {'value': 'present', 'comment': '...'}
            if 'value' in llm_value:
                # Recursively parse the extracted value
                return self._parse_llm_value(llm_value['value'])
            
            # Check for simple dictionary keys (e.g., {'present': 1})
            if 'present' in llm_value or 'endorsed' in llm_value:
                return 1
            elif 'conflict' in llm_value:
                return -1
            elif 'dominant' in llm_value:
                return 2
        
        # Try to parse dictionary-like strings
        if isinstance(llm_value, str) and '{' in llm_value:
            try:
                dict_val = eval(llm_value)
                if isinstance(dict_val, dict):
                    # Check for structured format first
                    if 'value' in dict_val:
                        return self._parse_llm_value(dict_val['value'])
                    
                    # Check for simple dictionary keys
                    if 'present' in dict_val or 'endorsed' in dict_val:
                        return 1
                    elif 'conflict' in dict_val:
                        return -1
                    elif 'dominant' in dict_val:
                        return 2
            except:
                pass
        
        return None
    
    def _parse_llm_response(self, response_text: str) -> Optional[Dict[str, int]]:
        """
        Parse LLM JSON response to extract 19 category annotations.
        
        The LLM returns annotations in a specific format with keys like
        "1_Value1_Self_Direction_Thought_values" and values like "present",
        "conflict", or None. This method converts them to the standard format.
        
        Args:
            response_text: Raw response text from LLM
        
        Returns:
            Dictionary mapping category names to annotation values (-1, 0, 1, 2),
            or None if parsing fails
        
        Requirements: 7.3
        """
        try:
            # Try to extract JSON from the response
            # The response might contain markdown code blocks or other text
            json_text = self._extract_json(response_text)
            
            if json_text is None:
                self.logger.error("Could not extract JSON from response")
                return None
            
            # Parse JSON
            response_data = json.loads(json_text)
            
            # Create case-insensitive lookup for response keys
            response_keys_lower = {k.lower(): k for k in response_data.keys()}
            
            # Convert to standard format
            predictions = {}
            
            for prompt_key, standard_key in self.CATEGORY_MAPPING.items():
                # Try exact match first
                if prompt_key in response_data:
                    llm_value = response_data[prompt_key]
                # Try case-insensitive match
                elif prompt_key.lower() in response_keys_lower:
                    actual_key = response_keys_lower[prompt_key.lower()]
                    llm_value = response_data[actual_key]
                else:
                    self.logger.warning(f"Missing category in response: {prompt_key}")
                    # Use default value of 0 (absent)
                    predictions[standard_key] = 0
                    continue
                
                # Try to parse the value in various formats
                parsed_value = self._parse_llm_value(llm_value)
                
                if parsed_value is not None:
                    predictions[standard_key] = parsed_value
                else:
                    self.logger.warning(
                        f"Unexpected value for {prompt_key}: {llm_value}, defaulting to 0"
                    )
                    predictions[standard_key] = 0
            
            return predictions
        
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {str(e)}")
            self.logger.debug(f"Response text: {response_text[:500]}")
            return None
        
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {str(e)}")
            return None
    
    def _extract_json(self, text: str) -> Optional[str]:
        """
        Extract JSON from response text.
        
        The LLM might wrap JSON in markdown code blocks or include
        additional text. This method tries to extract just the JSON.
        
        Args:
            text: Response text
        
        Returns:
            JSON string, or None if extraction fails
        """
        # Try to find JSON in markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()
        
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                json_candidate = text[start:end].strip()
                # Check if it starts with { or [
                if json_candidate.startswith('{') or json_candidate.startswith('['):
                    return json_candidate
        
        # Try to find JSON by looking for { ... }
        start = text.find('{')
        if start != -1:
            # Find the matching closing brace
            brace_count = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start:i+1]
        
        # If all else fails, try the entire text
        text = text.strip()
        if text.startswith('{') or text.startswith('['):
            return text
        
        return None
    
    @classmethod
    def load_predictions_from_csv(
        cls,
        csv_path: str,
        model_name: str = "gemini-csv"
    ) -> PredictionSet:
        """
        Load pre-computed predictions from a CSV file.
        
        This method allows loading Gemini predictions that were generated 
        previously and stored in a CSV file, rather than making live API calls.
        
        Expected CSV format:
        - Video ID column: 'filename' (format: @username_video_1234567890)
        - Value columns: Same 19 columns as ground truth (e.g., '1_Value1_Self_Direction_Thought_values')
        - Values: 'present', 'dominant', 'conflict', or empty
        - Extra columns like 'Has_sound' and 'Additional Notes' are ignored
        
        Args:
            csv_path: Path to the CSV file containing predictions
            model_name: Name to assign to this model (default: 'gemini-csv')
        
        Returns:
            PredictionSet containing all loaded predictions
        """
        logger = logging.getLogger(__name__)
        predictions: List[PredictionResult] = []
        failed_video_ids: List[str] = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        # Get video ID and normalize it
                        raw_video_id = row.get('filename', '').strip()
                        if not raw_video_id:
                            logger.warning(f"Skipping row with empty filename")
                            continue
                        
                        video_id = normalize_video_id(raw_video_id)
                        
                        # Parse value columns
                        prediction_values = {}
                        has_all_values = True
                        
                        for csv_column, standard_key in cls.CATEGORY_MAPPING.items():
                            if csv_column in row:
                                raw_value = row[csv_column].strip().lower() if row[csv_column] else ''
                                
                                # Map text value to integer
                                if raw_value in cls.VALUE_MAPPING:
                                    prediction_values[standard_key] = cls.VALUE_MAPPING[raw_value]
                                elif raw_value == '':
                                    prediction_values[standard_key] = 0
                                else:
                                    logger.warning(
                                        f"Unknown value '{raw_value}' for {csv_column} "
                                        f"in video {video_id}, defaulting to 0"
                                    )
                                    prediction_values[standard_key] = 0
                            else:
                                # Column not in CSV, this might be an error
                                logger.warning(
                                    f"Missing column {csv_column} in CSV for video {video_id}"
                                )
                                has_all_values = False
                        
                        # Check if we got all 19 values
                        if len(prediction_values) == 19:
                            predictions.append(PredictionResult(
                                video_id=video_id,
                                predictions=prediction_values,
                                success=True,
                                inference_time=0.0  # Pre-computed, no inference time
                            ))
                        else:
                            logger.warning(
                                f"Incomplete predictions for video {video_id}: "
                                f"got {len(prediction_values)}/19 values"
                            )
                            failed_video_ids.append(video_id)
                            predictions.append(PredictionResult(
                                video_id=video_id,
                                predictions=prediction_values,
                                success=False,
                                error_message=f"Incomplete: {len(prediction_values)}/19 values",
                                inference_time=0.0
                            ))
                    
                    except Exception as e:
                        raw_id = row.get('filename', 'unknown')
                        logger.error(f"Error parsing row for video {raw_id}: {str(e)}")
                        failed_video_ids.append(raw_id)
            
            # Build PredictionSet
            success_count = sum(1 for p in predictions if p.success)
            
            logger.info(
                f"Loaded {len(predictions)} predictions from {csv_path}, "
                f"{success_count} successful, {len(failed_video_ids)} failed"
            )
            
            return PredictionSet(
                model_name=model_name,
                predictions=predictions,
                total_count=len(predictions),
                success_count=success_count,
                failure_count=len(failed_video_ids),
                failed_video_ids=failed_video_ids
            )
        
        except FileNotFoundError:
            logger.error(f"Predictions CSV file not found: {csv_path}")
            return PredictionSet(
                model_name=model_name,
                predictions=[],
                total_count=0,
                success_count=0,
                failure_count=0,
                failed_video_ids=[]
            )
        
        except Exception as e:
            logger.error(f"Error loading predictions from CSV: {str(e)}", exc_info=True)
            return PredictionSet(
                model_name=model_name,
                predictions=[],
                total_count=0,
                success_count=0,
                failure_count=0,
                failed_video_ids=[]
            )
