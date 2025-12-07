# Tests for video ID normalization utilities

import pytest
from evaluation.video_id_utils import (
    normalize_video_id,
    extract_username,
    extract_video_number
)


class TestNormalizeVideoId:
    """Tests for normalize_video_id function."""
    
    def test_full_tiktok_url(self):
        """Test normalization of full TikTok URL."""
        url = "https://www.tiktok.com/@alexkay/video/6783398367490854150"
        expected = "alexkay_6783398367490854150"
        assert normalize_video_id(url) == expected
    
    def test_tiktok_url_without_www(self):
        """Test normalization of TikTok URL without www."""
        url = "https://tiktok.com/@alexkay/video/6783398367490854150"
        expected = "alexkay_6783398367490854150"
        assert normalize_video_id(url) == expected
    
    def test_tiktok_url_without_https(self):
        """Test normalization of TikTok URL without https."""
        url = "tiktok.com/@alexkay/video/6783398367490854150"
        expected = "alexkay_6783398367490854150"
        assert normalize_video_id(url) == expected
    
    def test_filename_format(self):
        """Test normalization of filename format (@username_video_ID)."""
        filename = "@alexkay_video_6783398367490854150"
        expected = "alexkay_6783398367490854150"
        assert normalize_video_id(filename) == expected
    
    def test_already_normalized(self):
        """Test that already normalized format is preserved."""
        normalized = "alexkay_6783398367490854150"
        assert normalize_video_id(normalized) == normalized
    
    def test_strips_whitespace(self):
        """Test that trailing whitespace is stripped."""
        url_with_space = "https://www.tiktok.com/@alexkay/video/6783398367490854150   "
        expected = "alexkay_6783398367490854150"
        assert normalize_video_id(url_with_space) == expected
    
    def test_leading_whitespace(self):
        """Test that leading whitespace is stripped."""
        url_with_space = "   @alexkay_video_6783398367490854150"
        expected = "alexkay_6783398367490854150"
        assert normalize_video_id(url_with_space) == expected
    
    def test_empty_string(self):
        """Test empty string returns empty string."""
        assert normalize_video_id("") == ""
    
    def test_none_returns_none(self):
        """Test None input returns None."""
        assert normalize_video_id(None) is None
    
    def test_unknown_format_returned_as_is(self):
        """Test that unknown format is returned as-is (stripped)."""
        unknown = "some_random_id  "
        assert normalize_video_id(unknown) == "some_random_id"
    
    def test_different_users(self):
        """Test normalization works for different usernames."""
        test_cases = [
            ("@user123_video_999999999", "user123_999999999"),
            ("https://www.tiktok.com/@test_user/video/12345", "test_user_12345"),
            ("test_user_12345", "test_user_12345"),
        ]
        for input_val, expected in test_cases:
            assert normalize_video_id(input_val) == expected


class TestExtractUsername:
    """Tests for extract_username function."""
    
    def test_from_tiktok_url(self):
        """Test username extraction from TikTok URL."""
        url = "https://www.tiktok.com/@alexkay/video/6783398367490854150"
        assert extract_username(url) == "alexkay"
    
    def test_from_filename(self):
        """Test username extraction from filename format."""
        filename = "@alexkay_video_6783398367490854150"
        assert extract_username(filename) == "alexkay"
    
    def test_from_normalized(self):
        """Test username extraction from normalized format."""
        normalized = "alexkay_6783398367490854150"
        assert extract_username(normalized) == "alexkay"
    
    def test_no_underscore_returns_none(self):
        """Test that ID without underscore returns None."""
        assert extract_username("nounderscore") is None


class TestExtractVideoNumber:
    """Tests for extract_video_number function."""
    
    def test_from_tiktok_url(self):
        """Test video number extraction from TikTok URL."""
        url = "https://www.tiktok.com/@alexkay/video/6783398367490854150"
        assert extract_video_number(url) == "6783398367490854150"
    
    def test_from_filename(self):
        """Test video number extraction from filename format."""
        filename = "@alexkay_video_6783398367490854150"
        assert extract_video_number(filename) == "6783398367490854150"
    
    def test_from_normalized(self):
        """Test video number extraction from normalized format."""
        normalized = "alexkay_6783398367490854150"
        assert extract_video_number(normalized) == "6783398367490854150"
    
    def test_no_underscore_returns_none(self):
        """Test that ID without underscore returns None."""
        assert extract_video_number("nounderscore") is None


class TestRealWorldExamples:
    """Test with real-world examples from the CSV files."""
    
    def test_ground_truth_to_prediction_matching(self):
        """Test that ground truth URLs and prediction filenames normalize to same ID."""
        # Ground truth format (full TikTok URL)
        ground_truth_url = "https://www.tiktok.com/@alexkay/video/6783398367490854150"
        
        # Prediction format (filename with @)
        prediction_filename = "@alexkay_video_6783398367490854150"
        
        # Both should normalize to the same ID
        gt_normalized = normalize_video_id(ground_truth_url)
        pred_normalized = normalize_video_id(prediction_filename)
        
        assert gt_normalized == pred_normalized
        assert gt_normalized == "alexkay_6783398367490854150"
    
    def test_whitespace_handling(self):
        """Test that trailing whitespace in CSV doesn't break matching."""
        # Ground truth often has trailing whitespace
        ground_truth_url = "https://www.tiktok.com/@alexkay/video/6783398367490854150   "
        prediction_filename = "@alexkay_video_6783398367490854150"
        
        assert normalize_video_id(ground_truth_url) == normalize_video_id(prediction_filename)
