# Script loading utility for model adapters

import os
import logging
from typing import Dict, Optional
from pathlib import Path


class ScriptLoader:
    """
    Utility for loading video scripts from GCS or local filesystem.
    
    Provides caching to avoid repeated reads of the same script file.
    Handles both GCS URIs (gs://) and local file paths.
    
    Requirements: 1.2, 2.4
    """
    
    def __init__(self):
        """Initialize the script loader with an empty cache."""
        self._cache: Dict[str, str] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._gcs_client = None
    
    def load_script(self, script_uri: str) -> Optional[str]:
        """
        Load script content from GCS or local filesystem.
        
        Uses caching to avoid repeated reads of the same file.
        
        Args:
            script_uri: Path to script file (GCS URI starting with 'gs://' or local path)
        
        Returns:
            Script content as string, or None if loading fails
        """
        # Check cache first
        if script_uri in self._cache:
            self.logger.debug(f"Returning cached script for {script_uri}")
            return self._cache[script_uri]
        
        try:
            # Determine if this is a GCS URI or local path
            if script_uri.startswith("gs://"):
                content = self._load_from_gcs(script_uri)
            else:
                content = self._load_from_local(script_uri)
            
            if content is not None:
                # Cache the loaded content
                self._cache[script_uri] = content
                self.logger.info(f"Successfully loaded script from {script_uri}")
            
            return content
        
        except Exception as e:
            self.logger.error(f"Error loading script from {script_uri}: {str(e)}")
            return None
    
    def _load_from_gcs(self, uri: str) -> Optional[str]:
        """
        Load script from Google Cloud Storage.
        
        Args:
            uri: GCS URI (e.g., 'gs://bucket/path/to/script.txt')
        
        Returns:
            Script content as string, or None if loading fails
        """
        try:
            # Lazy import and initialization of GCS client
            if self._gcs_client is None:
                from google.cloud import storage
                import os
                
                # Get project from environment variable
                project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
                if project_id:
                    self._gcs_client = storage.Client(project=project_id)
                else:
                    self._gcs_client = storage.Client()
            
            # Parse the URI to extract bucket and blob path
            if not uri.startswith("gs://"):
                raise ValueError(f"Invalid GCS URI format: {uri}")
            
            # Remove gs:// prefix and split bucket/path
            path_parts = uri[5:].split("/", 1)
            if len(path_parts) != 2:
                raise ValueError(f"Invalid GCS URI format: {uri}")
            
            bucket_name, blob_path = path_parts
            
            # Get the blob and download as text
            bucket = self._gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            if not blob.exists():
                self.logger.warning(f"Script file does not exist: {uri}")
                return None
            
            content = blob.download_as_text()
            return content
        
        except Exception as e:
            self.logger.error(f"Error loading script from GCS {uri}: {str(e)}")
            return None
    
    def _load_from_local(self, path: str) -> Optional[str]:
        """
        Load script from local filesystem.
        
        Args:
            path: Local file path
        
        Returns:
            Script content as string, or None if loading fails
        """
        try:
            file_path = Path(path)
            
            if not file_path.exists():
                self.logger.warning(f"Script file does not exist: {path}")
                return None
            
            if not file_path.is_file():
                self.logger.warning(f"Path is not a file: {path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return content
        
        except Exception as e:
            self.logger.error(f"Error loading script from local path {path}: {str(e)}")
            return None
    
    def clear_cache(self) -> None:
        """Clear the script cache."""
        self._cache.clear()
        self.logger.info("Script cache cleared")
    
    def get_cache_size(self) -> int:
        """
        Get the number of scripts currently cached.
        
        Returns:
            Number of cached scripts
        """
        return len(self._cache)
    
    def is_cached(self, script_uri: str) -> bool:
        """
        Check if a script is in the cache.
        
        Args:
            script_uri: Path to script file
        
        Returns:
            True if script is cached, False otherwise
        """
        return script_uri in self._cache
