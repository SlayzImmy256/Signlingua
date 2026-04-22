"""
Hugging Face Spaces Entry Point
This file is required for Hugging Face Spaces deployment
"""

# Import the Gradio app
from app.gradio_app import create_interface

# Create and launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
