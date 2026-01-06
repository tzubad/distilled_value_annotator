"""
Google Cloud Platform utility functions.
Handles credential cleanup and initialization.
"""

import os
import json
import logging


def cleanup_credentials():
    """
    Clean up Google Cloud credentials by stripping whitespace from project_id.
    This fixes issues where the service account JSON has trailing newlines.
    """
    try:
        creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if not creds_path or not os.path.exists(creds_path):
            return
        
        with open(creds_path, 'r') as f:
            creds_dict = json.load(f)
        
        # Check if project_id has whitespace
        if 'project_id' in creds_dict:
            original_id = creds_dict['project_id']
            cleaned_id = original_id.strip()
            
            if original_id != cleaned_id:
                creds_dict['project_id'] = cleaned_id
                with open(creds_path, 'w') as fw:
                    json.dump(creds_dict, fw, indent=2)
                logging.info(f"Cleaned up project_id in credentials: '{original_id}' -> '{cleaned_id}'")
    
    except Exception as e:
        logging.warning(f"Could not clean credentials: {e}")


# Clean up credentials when module is imported
cleanup_credentials()
