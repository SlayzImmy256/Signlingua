"""
Hugging Face Spaces Entry Point
This file is required for Hugging Face Spaces deployment
"""

import sys
import os
from pathlib import Path

# Print diagnostic information
print("="*60)
print("DIAGNOSTIC INFORMATION")
print("="*60)
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")
print(f"gradio_interface/ folder exists: {os.path.exists('gradio_interface')}")
print(f"src/ folder exists: {os.path.exists('src')}")
print(f"models/ folder exists: {os.path.exists('models')}")
print(f"models/best_model.pt exists: {os.path.exists('models/best_model.pt')}")
print("="*60)

try:
    # Import the Gradio app from gradio_interface folder
    from gradio_interface.gradio_app import create_interface
    print("✅ Successfully imported create_interface")
except Exception as e:
    print(f"❌ Error importing: {e}")
    import traceback
    traceback.print_exc()
    raise

# Create and launch the interface
if __name__ == "__main__":
    try:
        demo = create_interface()
        print("✅ Successfully created interface")
        demo.launch()
    except Exception as e:
        print(f"❌ Error launching: {e}")
        import traceback
        traceback.print_exc()
        raise
