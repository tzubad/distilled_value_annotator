"""
Video ID normalization utilities.

Handles different video ID formats:
1. Full TikTok URL: https://www.tiktok.com/@alexkay/video/6783398367490854150
2. Filename format: @alexkay_video_6783398367490854150

Both normalize to: alexkay_6783398367490854150
"""

import re
from typing import Optional


def normalize_video_id(video_id: str) -> str:
    """
    Normalize a video ID from various formats to a consistent format.
    
    Supported input formats:
    - Full TikTok URL: https://www.tiktok.com/@username/video/1234567890
    - Filename format: @username_video_1234567890
    - Already normalized: username_1234567890
    
    Output format: username_1234567890
    
    Args:
        video_id: The video ID in any supported format
        
    Returns:
        Normalized video ID in format: username_videoid
        
    Examples:
        >>> normalize_video_id("https://www.tiktok.com/@alexkay/video/6783398367490854150")
        'alexkay_6783398367490854150'
        >>> normalize_video_id("@alexkay_video_6783398367490854150")
        'alexkay_6783398367490854150'
        >>> normalize_video_id("alexkay_6783398367490854150")
        'alexkay_6783398367490854150'
    """
    if not video_id:
        return video_id
    
    # Strip whitespace (handles trailing spaces in CSVs)
    video_id = video_id.strip()
    
    # Pattern 1: Full TikTok URL
    # https://www.tiktok.com/@username/video/1234567890
    tiktok_url_pattern = r'(?:https?://)?(?:www\.)?tiktok\.com/@([^/]+)/video/(\d+)'
    match = re.search(tiktok_url_pattern, video_id)
    if match:
        username = match.group(1)
        video_num = match.group(2)
        return f"{username}_{video_num}"
    
    # Pattern 2: Filename format with @ prefix
    # @username_video_1234567890
    filename_pattern = r'^@([^_]+)_video_(\d+)$'
    match = re.match(filename_pattern, video_id)
    if match:
        username = match.group(1)
        video_num = match.group(2)
        return f"{username}_{video_num}"
    
    # Pattern 3: Already normalized format
    # username_1234567890
    normalized_pattern = r'^([^_]+)_(\d+)$'
    match = re.match(normalized_pattern, video_id)
    if match:
        # Already in normalized format
        return video_id
    
    # If no pattern matches, return as-is (with whitespace stripped)
    return video_id


def extract_username(video_id: str) -> Optional[str]:
    """
    Extract the username from a video ID.
    
    Args:
        video_id: The video ID in any supported format
        
    Returns:
        Username extracted from the video ID, or None if not extractable
    """
    normalized = normalize_video_id(video_id)
    if '_' in normalized:
        return normalized.split('_')[0]
    return None


def extract_video_number(video_id: str) -> Optional[str]:
    """
    Extract the numeric video ID from a video ID.
    
    Args:
        video_id: The video ID in any supported format
        
    Returns:
        Numeric video ID, or None if not extractable
    """
    normalized = normalize_video_id(video_id)
    if '_' in normalized:
        parts = normalized.split('_', 1)
        if len(parts) > 1:
            return parts[1]
    return None
