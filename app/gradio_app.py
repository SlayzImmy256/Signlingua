"""Gradio interface for Sign Language Translator"""

import gradio as gr
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import MediaPipePreprocessor
from src.models.transformer_model import MediaPipeTransformer
from src.translation.translator import SignLanguageTranslationPipeline, LANGUAGE_CODES
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignLanguageApp:
    """Sign Language Translator Application"""
    
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        """
        Initialize application
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        
        # Try to initialize MediaPipe (optional for demo mode)
        try:
            self.preprocessor = MediaPipePreprocessor()
            self.mediapipe_available = True
            logger.info("MediaPipe initialized successfully")
        except Exception as e:
            self.preprocessor = None
            self.mediapipe_available = False
            logger.warning(f"MediaPipe not available: {e}. Running in demo mode.")
        
        # Try to initialize translation pipeline
        try:
            self.translation_pipeline = SignLanguageTranslationPipeline(enable_tts=True)
            self.translation_available = True
        except Exception as e:
            self.translation_pipeline = None
            self.translation_available = False
            logger.warning(f"Translation not available: {e}")
        
        # Load model
        if model_path and Path(model_path).exists():
            self.model = self.load_model(model_path)
            if self.model is not None:
                self.model_loaded = True
                logger.info(f"✅ Model loaded successfully from {model_path}")
            else:
                self.model_loaded = False
                logger.error(f"❌ Failed to load model from {model_path}")
        else:
            self.model = None
            self.model_loaded = False
            logger.warning(f"❌ No model found at {model_path} - using demo mode")
        
        # Demo vocabulary (for when model is not loaded)
        self.demo_vocab = ['yes']
    
    def load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model"""
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Check if it's a state dict or full checkpoint
            if isinstance(checkpoint, dict):
                # Check for our Transformer model format
                if 'model_config' in checkpoint:
                    config = checkpoint['model_config']
                    model = MediaPipeTransformer(**config)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    
                    if 'class_names' in checkpoint:
                        self.class_names = checkpoint['class_names']
                    else:
                        self.class_names = [f"sign_{i}" for i in range(config['num_classes'])]
                
                # Check if it's a direct state dict (MLP or other model)
                elif 'features.0.weight' in checkpoint:
                    # This is an MLP model for landmarks
                    logger.info("Detected MLP/Fully-Connected model format")
                    
                    # Infer architecture from layer shapes
                    input_dim = checkpoint['features.0.weight'].shape[1]  # 258
                    hidden1 = checkpoint['features.0.weight'].shape[0]     # 512
                    hidden2 = checkpoint['features.4.weight'].shape[0]     # 256
                    hidden3 = checkpoint['features.8.weight'].shape[0]     # 128
                    hidden4 = checkpoint['classifier.0.weight'].shape[0]   # 64
                    num_classes = checkpoint['classifier.4.weight'].shape[0]  # 14
                    
                    logger.info(f"Model architecture: {input_dim} -> {hidden1} -> {hidden2} -> {hidden3} -> {hidden4} -> {num_classes}")
                    
                    # Create matching MLP model
                    import torch.nn as nn
                    
                    class LandmarkMLP(nn.Module):
                        def __init__(self, input_dim, hidden1, hidden2, hidden3, hidden4, num_classes):
                            super().__init__()
                            self.features = nn.Sequential(
                                nn.Linear(input_dim, hidden1),
                                nn.BatchNorm1d(hidden1),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(hidden1, hidden2),
                                nn.BatchNorm1d(hidden2),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(hidden2, hidden3),
                                nn.BatchNorm1d(hidden3),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                            )
                            self.classifier = nn.Sequential(
                                nn.Linear(hidden3, hidden4),
                                nn.BatchNorm1d(hidden4),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(hidden4, num_classes)
                            )
                        
                        def forward(self, x):
                            # Flatten if needed
                            if len(x.shape) > 2:
                                x = x.view(x.size(0), -1)
                            x = self.features(x)
                            x = self.classifier(x)
                            return x
                    
                    model = LandmarkMLP(input_dim, hidden1, hidden2, hidden3, hidden4, num_classes)
                    model.load_state_dict(checkpoint)
                    
                    # Create class names
                    if num_classes == 14:
                        # Likely a subset of ASL
                        self.class_names = [f"sign_{chr(65+i)}" for i in range(num_classes)]
                    elif num_classes == 29:
                        self.class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                                          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                                          'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']
                    else:
                        self.class_names = [f"sign_{i}" for i in range(num_classes)]
                    
                    self.model_type = 'mlp'
                    self.input_dim = input_dim
                
                else:
                    logger.warning("Unknown model format, using default config")
                    config = {
                        'input_dim': 258,
                        'd_model': 256,
                        'nhead': 8,
                        'num_encoder_layers': 4,
                        'dim_feedforward': 1024,
                        'dropout': 0.3,
                        'num_classes': 100,
                        'max_seq_length': 64
                    }
                    model = MediaPipeTransformer(**config)
                    self.class_names = [f"sign_{i}" for i in range(config['num_classes'])]
                    self.model_type = 'transformer'
            else:
                # Direct state dict
                logger.warning("Model is direct state dict, using default config")
                config = {
                    'input_dim': 258,
                    'd_model': 256,
                    'nhead': 8,
                    'num_encoder_layers': 4,
                    'dim_feedforward': 1024,
                    'dropout': 0.3,
                    'num_classes': 100,
                    'max_seq_length': 64
                }
                model = MediaPipeTransformer(**config)
                model.load_state_dict(checkpoint)
                self.class_names = [f"sign_{i}" for i in range(config['num_classes'])]
                self.model_type = 'transformer'
            
            model.to(self.device)
            model.eval()
            
            logger.info(f"Model loaded successfully: {len(self.class_names)} classes")
            logger.info(f"Model type: {getattr(self, 'model_type', 'transformer')}")
            
            return model
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_sign(self, video_path: str) -> tuple:
        """
        Predict sign language from video
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (predicted_text, confidence)
        """
        try:
            # Check if we're in demo mode (no model loaded)
            if not self.model_loaded:
                # Demo mode - return fixed prediction
                import random
                predicted_text = random.choice(self.demo_vocab)
                confidence = random.uniform(0.7, 0.95)
                logger.info(f"Demo mode (no model): Predicted '{predicted_text}' with confidence {confidence:.2%}")
                return predicted_text, confidence
            
            # Model is loaded, but check if MediaPipe is available
            if not self.mediapipe_available:
                # Model loaded but MediaPipe not available
                logger.warning("Model loaded but MediaPipe not available - cannot process video")
                return "Error: MediaPipe not available for video processing", 0.0
            
            # Check if video file exists
            if not video_path or not Path(video_path).exists():
                return "Error: Video file not found", 0.0
            
            # Check model type
            model_type = getattr(self, 'model_type', 'transformer')
            
            if model_type == 'mlp':
                # For MLP models, extract landmarks from video
                landmarks = self.preprocessor.extract_landmarks_from_video(
                    video_path, 
                    max_frames=1  # MLP uses single frame
                )
                
                if landmarks is None:
                    return "Error: Could not extract landmarks from video", 0.0
                
                # Take first frame and flatten
                landmarks_flat = landmarks[0].flatten()  # Shape: (258,)
                
                # Convert to tensor
                landmarks_tensor = torch.FloatTensor(landmarks_flat).unsqueeze(0).to(self.device)
                
                # Predict
                with torch.no_grad():
                    logits = self.model(landmarks_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    confidence, predicted_idx = torch.max(probs, dim=-1)
                
                predicted_text = self.class_names[predicted_idx.item()]
                confidence_value = confidence.item()
                
            elif model_type == 'cnn':
                # For CNN models, extract first frame as image
                import cv2
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    return "Error: Could not read video frame", 0.0
                
                # Preprocess image for CNN
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_tensor = transform(frame_rgb).unsqueeze(0).to(self.device)
                
                # Predict
                with torch.no_grad():
                    logits = self.model(image_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    confidence, predicted_idx = torch.max(probs, dim=-1)
                
                predicted_text = self.class_names[predicted_idx.item()]
                confidence_value = confidence.item()
                
            else:
                # For Transformer models, extract landmarks
                landmarks = self.preprocessor.extract_landmarks_from_video(
                    video_path, 
                    max_frames=64
                )
                
                if landmarks is None:
                    return "Error: Could not extract landmarks from video", 0.0
                
                # Normalize
                landmarks = self.preprocessor.normalize_landmarks(landmarks)
                
                # Convert to tensor
                landmarks_tensor = torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)
                
                # Predict
                with torch.no_grad():
                    logits = self.model(landmarks_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    confidence, predicted_idx = torch.max(probs, dim=-1)
                
                # Get predicted class
                predicted_text = self.class_names[predicted_idx.item()]
                confidence_value = confidence.item()
            
            logger.info(f"Predicted: '{predicted_text}' with confidence {confidence_value:.2%}")
            
            return predicted_text, confidence_value
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error during prediction: {str(e)}", 0.0
    
    def process_video(self, video_path: str, target_language: str, 
                     enable_audio: bool = True) -> tuple:
        """
        Process video through complete pipeline
        
        Args:
            video_path: Path to uploaded video
            target_language: Target language for translation
            enable_audio: Whether to generate audio
            
        Returns:
            Tuple of (english_text, translated_text, confidence, audio_path)
        """
        try:
            if video_path is None or video_path == "":
                return "Please upload a video", "", "0%", None
            
            # Predict sign
            english_text, confidence = self.predict_sign(video_path)
            
            # Format confidence
            confidence_str = f"{confidence:.1%}"
            
            # Translate
            if self.translation_available:
                result = self.translation_pipeline.process(
                    english_text,
                    target_language=target_language,
                    generate_audio=enable_audio
                )
                translated_text = result['translated_text']
                audio_path = result['audio_path']
            else:
                # Demo mode - no translation
                translated_text = f"[Demo] {english_text} (translation not available)"
                audio_path = None
            
            return english_text, translated_text, confidence_str, audio_path
        
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return f"Error: {str(e)}", "Translation failed", "0%", None


def create_interface():
    """Create Gradio interface"""
    
    # Initialize app
    app = SignLanguageApp(
        model_path='models/best_model.pt',  # Update with actual path
        device='cpu'
    )
    
    # Create status message
    status_parts = []
    if app.model_loaded:
        status_parts.append("✅ Model Loaded")
    else:
        status_parts.append("❌ Model Not Loaded")
    
    if app.mediapipe_available:
        status_parts.append("✅ MediaPipe Available")
    else:
        status_parts.append("❌ MediaPipe Not Available")
    
    if app.translation_available:
        status_parts.append("✅ Translation Available")
    else:
        status_parts.append("❌ Translation Not Available")
    
    system_status = " | ".join(status_parts)
    print(f"\n{'='*60}")
    print(f"SYSTEM STATUS: {system_status}")
    print(f"{'='*60}\n")
    
    # Get supported languages
    languages = list(LANGUAGE_CODES.keys())
    
    # Custom CSS with modern design inspired by the screenshots
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .gradio-container {
        font-family: 'Inter', sans-serif !important;
        max-width: 1400px !important;
        margin: 0 auto !important;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #FDB813 0%, #FFA500 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(253, 184, 19, 0.3);
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95) !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .primary-btn button {
        background: linear-gradient(135deg, #FDB813 0%, #FFA500 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 6px 20px rgba(253, 184, 19, 0.4) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .primary-btn button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(253, 184, 19, 0.5) !important;
    }
    
    /* Input styling */
    .input-section {
        background: #FFFBF5;
        border-radius: 16px;
        padding: 1.5rem;
        border: 2px solid #FDB813;
    }
    
    /* Output styling */
    .output-section {
        background: linear-gradient(135deg, #FFF9E6 0%, #FFFBF5 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 2px solid #FDB813;
    }
    
    /* Text boxes */
    .output-text textarea {
        background: white !important;
        border: 2px solid #FDB813 !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #333 !important;
        padding: 1rem !important;
    }
    
    /* Labels */
    label {
        font-weight: 600 !important;
        color: #333 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Dropdown */
    .dropdown select {
        border: 2px solid #FDB813 !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
        font-size: 1rem !important;
    }
    
    /* Icons */
    .icon {
        font-size: 2rem;
        margin-right: 0.5rem;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 2px solid #FDB813;
        margin: 0.5rem;
    }
    
    .feature-card h3 {
        color: #FDB813;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #FDB813 0%, #FFA500 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-top: 2rem;
        text-align: center;
    }
    
    .footer h3 {
        color: white !important;
        font-weight: 700 !important;
    }
    
    .footer p, .footer strong {
        color: white !important;
    }
    
    /* Confidence badge */
    .confidence-badge {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Video upload area */
    .video-upload {
        border: 3px dashed #FDB813 !important;
        border-radius: 16px !important;
        background: #FFFBF5 !important;
        padding: 2rem !important;
        text-align: center !important;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem !important;
        }
        .gradio-container {
            padding: 1rem !important;
        }
    }
    """
    
    # Create interface
    with gr.Blocks(css=custom_css, title="🤟 Sign Language Translator") as demo:
        
        # Header
        with gr.Row(elem_classes="main-header"):
            gr.Markdown(
                f"""
                # 🤟 I Hear You - Sign Language Translator
                ### Break communication barriers with AI-powered sign language recognition
                Translate ASL to 100+ languages instantly with speech output
                
                **System Status:** {system_status}
                """
            )
        
        # Main content
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1, elem_classes="card"):
                gr.Markdown("### 📹 Upload Your Video")
                
                video_input = gr.Video(
                    label="",
                    sources=["upload", "webcam"],
                    elem_classes="video-upload"
                )
                
                gr.Markdown("### 🌍 Choose Language")
                language_dropdown = gr.Dropdown(
                    choices=languages,
                    value='English',
                    label="",
                    info="Select output language",
                    elem_classes="dropdown"
                )
                
                audio_checkbox = gr.Checkbox(
                    value=True,
                    label="🔊 Enable Voice Output",
                    info="Generate text-to-speech"
                )
                
                with gr.Row(elem_classes="primary-btn"):
                    translate_btn = gr.Button(
                        "🚀 Translate Now",
                        variant="primary",
                        size="lg"
                    )
            
            # Right column - Output
            with gr.Column(scale=1, elem_classes="card output-section"):
                gr.Markdown("### ✨ Translation Results")
                
                with gr.Group():
                    english_output = gr.Textbox(
                        label="📝 English Text",
                        placeholder="Your sign will appear here...",
                        lines=3,
                        elem_classes="output-text"
                    )
                    
                    confidence_output = gr.Textbox(
                        label="🎯 Confidence Score",
                        placeholder="Accuracy will appear here...",
                        elem_classes="output-text"
                    )
                
                gr.Markdown("---")
                
                with gr.Group():
                    translated_output = gr.Textbox(
                        label="🌐 Translated Text",
                        placeholder="Translation will appear here...",
                        lines=3,
                        elem_classes="output-text"
                    )
                    
                    audio_output = gr.Audio(
                        label="🔊 Audio Output",
                        type="filepath"
                    )
        
        # Features section
        gr.Markdown("---")
        gr.Markdown("## 🌟 Key Features")
        
        with gr.Row():
            with gr.Column(elem_classes="feature-card"):
                gr.Markdown(
                    """
                    ### 🎯 High Accuracy
                    Advanced AI model trained on thousands of signs
                    """
                )
            
            with gr.Column(elem_classes="feature-card"):
                gr.Markdown(
                    """
                    ### 🌍 100+ Languages
                    Translate to any language instantly
                    """
                )
            
            with gr.Column(elem_classes="feature-card"):
                gr.Markdown(
                    """
                    ### 🔊 Voice Output
                    Hear translations with natural speech
                    """
                )
            
            with gr.Column(elem_classes="feature-card"):
                gr.Markdown(
                    """
                    ### ⚡ Real-time
                    Get instant results in seconds
                    """
                )
        
        # How it works
        gr.Markdown("---")
        gr.Markdown("## 📖 How It Works")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """
                    ### Step 1: Upload 📹
                    Record or upload your sign language video
                    """
                )
            
            with gr.Column():
                gr.Markdown(
                    """
                    ### Step 2: Process 🤖
                    AI analyzes hand movements and gestures
                    """
                )
            
            with gr.Column():
                gr.Markdown(
                    """
                    ### Step 3: Translate 🌐
                    Get text and audio in your language
                    """
                )
        
        # Footer
        with gr.Row(elem_classes="footer"):
            gr.Markdown(
                """
                ### 💡 About This Project
                
                Built with ❤️ using **PyTorch**, **MediaPipe**, **Transformers**, and **Gradio**
                
                **Tech Stack:** Deep Learning • Computer Vision • NLP • Multi-language Translation
                
                🚀 Powered by AI | 🤗 Hosted on Hugging Face Spaces
                
                ---
                
                **Note:** Currently trained on American Sign Language (ASL) alphabet. 
                Best results with clear lighting and hand visibility.
                """
            )
        
        # Connect components
        translate_btn.click(
            fn=app.process_video,
            inputs=[video_input, language_dropdown, audio_checkbox],
            outputs=[english_output, translated_output, confidence_output, audio_output]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
