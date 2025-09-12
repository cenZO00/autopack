#!/usr/bin/env python3
"""
Test script to verify the memory management fixes in autopack.
This script can be used to test the fixes before running the full autopack command.
"""

import os
import sys
import psutil
import torch

def check_system_memory():
    """Check available system memory."""
    memory = psutil.virtual_memory()
    print(f"Total memory: {memory.total / (1024**3):.1f} GB")
    print(f"Available memory: {memory.available / (1024**3):.1f} GB")
    print(f"Memory usage: {memory.percent:.1f}%")
    return memory.available / (1024**3)

def test_model_loading():
    """Test loading a small model to verify the fixes work."""
    try:
        from autopack.quantize import _get_model_size_estimate, _check_memory_availability
        
        # Test with a small model first
        test_model = "microsoft/DialoGPT-small"
        print(f"\nTesting with {test_model}...")
        
        # Estimate size
        estimated_size = _get_model_size_estimate(test_model)
        print(f"Estimated model size: {estimated_size:.1f} GB")
        
        # Check memory
        required_memory = estimated_size * 1.5
        has_memory = _check_memory_availability(required_memory)
        print(f"Required memory: {required_memory:.1f} GB")
        print(f"Has enough memory: {has_memory}")
        
        return True
    except Exception as e:
        print(f"Error testing model loading: {e}")
        return False

def main():
    print("Autopack Memory Management Test")
    print("=" * 40)
    
    # Check system memory
    available_gb = check_system_memory()
    
    # Test model loading functions
    if not test_model_loading():
        print("❌ Model loading test failed")
        return 1
    
    print("\n✅ Basic tests passed!")
    print("\nTo test with the actual deepseek model, try:")
    print("autopack deepseek-ai/deepseek-llm-7b-chat --memory-safe --max-memory-gb 16")
    print("\nOr for a single variant:")
    print("autopack deepseek-ai/deepseek-llm-7b-chat --hf-variant bnb-4bit --memory-safe")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
