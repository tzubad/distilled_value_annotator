# Unit tests for ScriptLoader utility

import pytest
import tempfile
import os
from pathlib import Path
from evaluation.adapters import ScriptLoader


def test_script_loader_loads_local_file():
    """Test that ScriptLoader can load a local file."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test script content")
        temp_path = f.name
    
    try:
        loader = ScriptLoader()
        content = loader.load_script(temp_path)
        
        assert content is not None
        assert content == "Test script content"
    finally:
        os.unlink(temp_path)


def test_script_loader_caches_content():
    """Test that ScriptLoader caches loaded scripts."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Cached content")
        temp_path = f.name
    
    try:
        loader = ScriptLoader()
        
        # First load
        assert loader.get_cache_size() == 0
        content1 = loader.load_script(temp_path)
        assert loader.get_cache_size() == 1
        assert loader.is_cached(temp_path)
        
        # Second load (should use cache)
        content2 = loader.load_script(temp_path)
        assert loader.get_cache_size() == 1
        assert content1 == content2
    finally:
        os.unlink(temp_path)


def test_script_loader_handles_missing_file():
    """Test that ScriptLoader handles missing files gracefully."""
    loader = ScriptLoader()
    content = loader.load_script("/nonexistent/path/to/file.txt")
    
    assert content is None


def test_script_loader_clear_cache():
    """Test that ScriptLoader can clear its cache."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content")
        temp_path = f.name
    
    try:
        loader = ScriptLoader()
        
        # Load and cache
        loader.load_script(temp_path)
        assert loader.get_cache_size() == 1
        
        # Clear cache
        loader.clear_cache()
        assert loader.get_cache_size() == 0
        assert not loader.is_cached(temp_path)
    finally:
        os.unlink(temp_path)


def test_script_loader_handles_multiple_files():
    """Test that ScriptLoader can handle multiple files."""
    files = []
    
    try:
        # Create multiple temporary files
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(f"Content {i}")
                files.append(f.name)
        
        loader = ScriptLoader()
        
        # Load all files
        for path in files:
            content = loader.load_script(path)
            assert content is not None
        
        # Verify all are cached
        assert loader.get_cache_size() == 3
        for path in files:
            assert loader.is_cached(path)
    
    finally:
        # Clean up
        for path in files:
            os.unlink(path)


def test_script_loader_handles_unicode():
    """Test that ScriptLoader handles Unicode content correctly."""
    # Create a file with Unicode content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write("Test with émojis 🎬 and spëcial çharacters")
        temp_path = f.name
    
    try:
        loader = ScriptLoader()
        content = loader.load_script(temp_path)
        
        assert content is not None
        assert "émojis" in content
        assert "🎬" in content
        assert "spëcial" in content
    finally:
        os.unlink(temp_path)
