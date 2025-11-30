"""
Unit tests for GCS interface module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from gcs import GCSInterface


class TestGCSInterface:
    """Test suite for GCSInterface class."""
    
    @patch('gcs.storage.Client')
    def test_initialization(self, mock_client_class):
        """Test GCSInterface initialization."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client
        
        gcs = GCSInterface('test-bucket')
        
        assert gcs.bucket_name == 'test-bucket'
        assert gcs.client == mock_client
        assert gcs.bucket == mock_bucket
        mock_client.bucket.assert_called_once_with('test-bucket')
    
    @patch('gcs.storage.Client')
    def test_list_videos_success(self, mock_client_class):
        """Test listing video files from GCS."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Create mock blobs
        mock_blob1 = Mock()
        mock_blob1.name = 'videos/video1.mp4'
        mock_blob2 = Mock()
        mock_blob2.name = 'videos/video2.MP4'
        mock_blob3 = Mock()
        mock_blob3.name = 'videos/script.txt'  # Not a video
        
        mock_client.list_blobs.return_value = [mock_blob1, mock_blob2, mock_blob3]
        
        gcs = GCSInterface('test-bucket')
        videos = gcs.list_videos('videos/')
        
        assert len(videos) == 2
        assert 'gs://test-bucket/videos/video1.mp4' in videos
        assert 'gs://test-bucket/videos/video2.MP4' in videos
        mock_client.list_blobs.assert_called_once_with('test-bucket', prefix='videos/')
    
    @patch('gcs.storage.Client')
    def test_list_videos_empty(self, mock_client_class):
        """Test listing videos when no videos exist."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.list_blobs.return_value = []
        
        gcs = GCSInterface('test-bucket')
        videos = gcs.list_videos('videos/')
        
        assert videos == []
    
    @patch('gcs.storage.Client')
    def test_list_videos_error(self, mock_client_class):
        """Test error handling when listing videos fails."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.list_blobs.side_effect = Exception('GCS error')
        
        gcs = GCSInterface('test-bucket')
        
        with pytest.raises(Exception, match='GCS error'):
            gcs.list_videos('videos/')
    
    @patch('gcs.storage.Client')
    def test_list_scripts_success(self, mock_client_class):
        """Test listing script files from GCS."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Create mock blobs
        mock_blob1 = Mock()
        mock_blob1.name = 'scripts/script1.txt'
        mock_blob2 = Mock()
        mock_blob2.name = 'scripts/script2.TXT'
        mock_blob3 = Mock()
        mock_blob3.name = 'scripts/video.mp4'  # Not a script
        
        mock_client.list_blobs.return_value = [mock_blob1, mock_blob2, mock_blob3]
        
        gcs = GCSInterface('test-bucket')
        scripts = gcs.list_scripts('scripts/')
        
        assert len(scripts) == 2
        assert 'gs://test-bucket/scripts/script1.txt' in scripts
        assert 'gs://test-bucket/scripts/script2.TXT' in scripts
    
    @patch('gcs.storage.Client')
    def test_read_script_success(self, mock_client_class):
        """Test reading script content from GCS."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_blob.download_as_text.return_value = 'Script content here'
        mock_bucket.blob.return_value = mock_blob
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client
        
        gcs = GCSInterface('test-bucket')
        content = gcs.read_script('gs://test-bucket/scripts/script1.txt')
        
        assert content == 'Script content here'
        mock_client.bucket.assert_called_with('test-bucket')
        mock_bucket.blob.assert_called_with('scripts/script1.txt')
        mock_blob.download_as_text.assert_called_once()
    
    @patch('gcs.storage.Client')
    def test_read_script_invalid_uri(self, mock_client_class):
        """Test error handling for invalid GCS URI."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        gcs = GCSInterface('test-bucket')
        
        with pytest.raises(ValueError, match='Invalid GCS URI format'):
            gcs.read_script('invalid-uri')
        
        with pytest.raises(ValueError, match='Invalid GCS URI format'):
            gcs.read_script('gs://bucket-only')
    
    @patch('gcs.storage.Client')
    def test_read_script_error(self, mock_client_class):
        """Test error handling when reading script fails."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_blob.download_as_text.side_effect = Exception('Download failed')
        mock_bucket.blob.return_value = mock_blob
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client
        
        gcs = GCSInterface('test-bucket')
        
        with pytest.raises(Exception, match='Download failed'):
            gcs.read_script('gs://test-bucket/scripts/script1.txt')
    
    @patch('gcs.storage.Client')
    def test_save_script_success(self, mock_client_class):
        """Test saving script content to GCS."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_bucket.blob.return_value = mock_blob
        mock_client_class.return_value = mock_client
        
        gcs = GCSInterface('test-bucket')
        gcs.bucket = mock_bucket
        
        result = gcs.save_script('Script content', 'scripts/output.txt')
        
        assert result is True
        mock_bucket.blob.assert_called_once_with('scripts/output.txt')
        mock_blob.upload_from_string.assert_called_once_with('Script content', content_type='text/plain')
    
    @patch('gcs.storage.Client')
    def test_save_script_error(self, mock_client_class):
        """Test error handling when saving script fails."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_blob.upload_from_string.side_effect = Exception('Upload failed')
        mock_bucket.blob.return_value = mock_blob
        mock_client_class.return_value = mock_client
        
        gcs = GCSInterface('test-bucket')
        gcs.bucket = mock_bucket
        
        result = gcs.save_script('Script content', 'scripts/output.txt')
        
        assert result is False
    
    @patch('gcs.storage.Client')
    def test_save_csv_success(self, mock_client_class):
        """Test saving CSV content to GCS."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_bucket.blob.return_value = mock_blob
        mock_client_class.return_value = mock_client
        
        gcs = GCSInterface('test-bucket')
        gcs.bucket = mock_bucket
        
        csv_content = 'video_id,value1,value2\nv1,1,2\n'
        result = gcs.save_csv(csv_content, 'output/results.csv')
        
        assert result is True
        mock_bucket.blob.assert_called_once_with('output/results.csv')
        mock_blob.upload_from_string.assert_called_once_with(csv_content, content_type='text/csv')
    
    @patch('gcs.storage.Client')
    def test_save_csv_error(self, mock_client_class):
        """Test error handling when saving CSV fails."""
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_blob.upload_from_string.side_effect = Exception('Upload failed')
        mock_bucket.blob.return_value = mock_blob
        mock_client_class.return_value = mock_client
        
        gcs = GCSInterface('test-bucket')
        gcs.bucket = mock_bucket
        
        result = gcs.save_csv('csv content', 'output/results.csv')
        
        assert result is False
