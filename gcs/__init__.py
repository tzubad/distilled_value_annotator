# GCS interface module for cloud storage operations

from google.cloud import storage
from typing import List, Optional
import logging


class GCSInterface:
    """
    Interface for Google Cloud Storage operations.
    Handles reading and writing videos, scripts, and CSV files.
    """
    
    def __init__(self, bucket_name: str):
        """
        Initialize GCS interface with bucket name.
        
        Args:
            bucket_name: Name of the GCS bucket to use
        """
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        logging.info(f"GCSInterface initialized for bucket: {bucket_name}")
    
    def list_videos(self, prefix: str) -> List[str]:
        """
        List all MP4 video files in the specified GCS path.
        
        Args:
            prefix: Path prefix to search for videos (e.g., "movies/videos/")
        
        Returns:
            List of GCS URIs for MP4 files
        """
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            video_uris = []
            
            for blob in blobs:
                if blob.name.lower().endswith('.mp4'):
                    uri = f"gs://{self.bucket_name}/{blob.name}"
                    video_uris.append(uri)
            
            logging.info(f"Found {len(video_uris)} video files in {prefix}")
            return video_uris
        
        except Exception as e:
            logging.error(f"Error listing videos from {prefix}: {str(e)}")
            raise
    
    def list_scripts(self, prefix: str) -> List[str]:
        """
        List all text script files in the specified GCS path.
        
        Args:
            prefix: Path prefix to search for scripts (e.g., "movies/scripts/")
        
        Returns:
            List of GCS URIs for text files
        """
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            script_uris = []
            
            for blob in blobs:
                if blob.name.lower().endswith('.txt'):
                    uri = f"gs://{self.bucket_name}/{blob.name}"
                    script_uris.append(uri)
            
            logging.info(f"Found {len(script_uris)} script files in {prefix}")
            return script_uris
        
        except Exception as e:
            logging.error(f"Error listing scripts from {prefix}: {str(e)}")
            raise
    
    def read_script(self, uri: str) -> str:
        """
        Read script content from GCS.
        
        Args:
            uri: GCS URI of the script file (e.g., "gs://bucket/path/file.txt")
        
        Returns:
            Content of the script file as string
        """
        try:
            # Parse the URI to extract bucket and blob path
            if not uri.startswith("gs://"):
                raise ValueError(f"Invalid GCS URI format: {uri}")
            
            # Remove gs:// prefix and split bucket/path
            path_parts = uri[5:].split("/", 1)
            if len(path_parts) != 2:
                raise ValueError(f"Invalid GCS URI format: {uri}")
            
            bucket_name, blob_path = path_parts
            
            # Get the blob and download as text
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            content = blob.download_as_text()
            
            logging.info(f"Successfully read script from {uri}")
            return content
        
        except Exception as e:
            logging.error(f"Error reading script from {uri}: {str(e)}")
            raise
    
    def save_script(self, content: str, path: str) -> bool:
        """
        Save script content to GCS.
        
        Args:
            content: Script content to save
            path: GCS path where to save the script (relative to bucket)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            blob = self.bucket.blob(path)
            blob.upload_from_string(content, content_type='text/plain')
            
            logging.info(f"Successfully saved script to gs://{self.bucket_name}/{path}")
            return True
        
        except Exception as e:
            logging.error(f"Error saving script to {path}: {str(e)}")
            return False
    
    def save_csv(self, csv_content: str, path: str) -> bool:
        """
        Save CSV content to GCS.
        
        Args:
            csv_content: CSV content to save
            path: GCS path where to save the CSV (relative to bucket)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            blob = self.bucket.blob(path)
            blob.upload_from_string(csv_content, content_type='text/csv')
            
            logging.info(f"Successfully saved CSV to gs://{self.bucket_name}/{path}")
            return True
        
        except Exception as e:
            logging.error(f"Error saving CSV to {path}: {str(e)}")
            return False
