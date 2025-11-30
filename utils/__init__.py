# Utility functions and helpers

import pandas as pd
import logging
from typing import List, Dict
from gcs import GCSInterface
from utils.logger import PipelineLogger


class CSVGenerator:
    """
    Generator for creating CSV files from annotation data.
    Handles conversion of annotations to DataFrame and upload to GCS.
    """
    
    def __init__(self, gcs_interface: GCSInterface):
        """
        Initialize the CSV generator.
        
        Args:
            gcs_interface: GCSInterface instance for uploading CSV to GCS
        """
        self.gcs_interface = gcs_interface
        logging.info("CSVGenerator initialized")

    
    def generate_and_save(self, annotations: List[Dict], output_path: str) -> bool:
        """
        Generate CSV from annotations and upload to GCS.
        
        Args:
            annotations: List of annotation dictionaries containing video_id and value scores
            output_path: GCS path where to save the CSV (relative to bucket)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not annotations:
                logging.warning("No annotations provided, creating empty CSV")
                return False
            
            logging.info(f"Generating CSV from {len(annotations)} annotations")
            
            # Normalize incoming annotation keys to match expected CSV columns
            # Incoming format: '1_Value1_Self_Direction_Action_values': 'present'
            # Target format: 'Self_Direction_Action': 'present'
            normalized_annotations = []
            
            # Variant mapping for label inconsistencies
            variant_to_target = {
                'Power_dominance': 'Power_Dominance',
                'Benevolence_Caring': 'Benevolence_Care',
            }
            
            import re
            
            for ann in annotations:
                normalized = {}
                
                # Preserve metadata fields
                for meta_key in ('video_id', 'notes', 'Has_sound'):
                    if meta_key in ann:
                        if meta_key == 'Has_sound':
                            val = ann[meta_key]
                            if isinstance(val, str):
                                normalized[meta_key] = val.strip().lower() == 'true'
                            else:
                                normalized[meta_key] = val
                        else:
                            normalized[meta_key] = ann[meta_key]
                
                # Parse value keys: '1_Value1_{label}_values'
                for key, value in ann.items():
                    m = re.match(r'^\d+_Value\d+_(?P<label>[A-Za-z_]+)_values$', key, re.IGNORECASE)
                    if m:
                        raw_label = m.group('label')
                        target_label = variant_to_target.get(raw_label, raw_label)
                        normalized[target_label] = value
                
                normalized_annotations.append(normalized)
            
            # Convert normalized annotations list to pandas DataFrame
            df = pd.DataFrame(normalized_annotations)
            
            # Define the correct column ordering
            # video_id, 19 value columns, has_sound, notes
            column_order = [
                'video_id',
                'Self_Direction_Thought',
                'Self_Direction_Action',
                'Stimulation',
                'Hedonism',
                'Achievement',
                'Power_Resources',
                'Power_Dominance',
                'Face',
                'Security_Personal',
                'Security_Social',
                'Conformity_Rules',
                'Conformity_Interpersonal',
                'Tradition',
                'Humility',
                'Benevolence_Dependability',
                'Benevolence_Care',
                'Universalism_Concern',
                'Universalism_Nature',
                'Universalism_Tolerance',
                'Has_sound',
                'notes'
            ]
            
            # Ensure all expected columns exist, add missing ones with None
            for col in column_order:
                if col not in df.columns:
                    df[col] = None
                    logging.warning(f"Column '{col}' not found in annotations, adding with None values")
            
            # Reorder columns according to specification
            df = df[column_order]
            
            # Convert DataFrame to CSV string
            csv_content = df.to_csv(index=False)
            
            logging.info(f"CSV generated with {len(df)} rows and {len(df.columns)} columns")
            
            # Upload to GCS using GCSInterface
            success = self.gcs_interface.save_csv(csv_content, output_path)
            
            if success:
                logging.info(f"CSV successfully uploaded to gs://{self.gcs_interface.bucket_name}/{output_path}")
            else:
                logging.error(f"Failed to upload CSV to {output_path}")
            
            return success
        
        except Exception as e:
            logging.error(f"Error generating and saving CSV: {str(e)}")
            return False
