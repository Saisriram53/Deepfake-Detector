"""
Streamlit Web Interface for Deepfake Detection
Simple interface compatible with EfficientNet-B0 model
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import tempfile
import time

# Model Definition - EfficientNet-B0
class LightweightDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 2)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Configuration
MODEL_PATH = Path("models/best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    """Load the trained model"""
    model = LightweightDeepfakeDetector()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict_image(model, image):
    """Predict if an image is real or fake"""
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()

def process_video(model, video_path, frame_skip=5):
    """Process video and return predictions"""
    cap = cv2.VideoCapture(str(video_path))
    
    frame_count = 0
    predictions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for faster processing
        if frame_count % frame_skip == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred, conf, probs = predict_image(model, frame_rgb)
            predictions.append({
                'frame': frame_count,
                'prediction': pred,
                'confidence': conf,
                'probabilities': probs
            })
        
        frame_count += 1
    
    cap.release()
    return predictions

# Page config
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .real-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .fake-badge {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔍 Deepfake Detection System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This system uses a **EfficientNet-B0** model trained on 16,433 images 
    to detect deepfake videos and images.
    
    **Model Performance:**
    - Test Accuracy: 94%
    - Validation Accuracy: 95.04%
    - Model Size: 15.3 MB
    """)
    
    st.header("📊 Model Info")
    if MODEL_PATH.exists():
        st.success("✅ Model loaded")
        st.info(f"💾 {MODEL_PATH}")
        st.info(f"🖥️ Device: {DEVICE}")
    else:
        st.error("❌ Model not found!")
    
    st.header("🎯 Usage")
    st.write("""
    1. Upload an image or video
    2. Wait for processing
    3. View the prediction results
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📸 Image Detection", "🎥 Video Detection", "📈 Statistics"])

with tab1:
    st.header("Image Deepfake Detection")
    
    uploaded_file = st.file_uploader(
        "Upload an image (JPG, PNG, JPEG)",
        type=['jpg', 'jpeg', 'png'],
        key='image_uploader'
    )
    
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Predict
        with st.spinner("🔍 Analyzing image..."):
            try:
                model = load_model()
                pred_class, confidence, probs = predict_image(model, image)
                
                with col2:
                    st.subheader("Analysis Result")
                    
                    # Result badge
                    if pred_class == 0:
                        st.markdown('<div class="fake-badge">⚠️ FAKE DETECTED</div>', unsafe_allow_html=True)
                        st.error("This image appears to be AI-generated or manipulated.")
                    else:
                        st.markdown('<div class="real-badge">✅ REAL</div>', unsafe_allow_html=True)
                        st.success("This image appears to be authentic.")
                    
                    st.write("")
                    
                    # Confidence metrics
                    st.metric("Confidence", f"{confidence:.2f}%")
                    
                    # Probability bars
                    st.write("**Probability Distribution:**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Fake", f"{probs[0]*100:.1f}%")
                    with col_b:
                        st.metric("Real", f"{probs[1]*100:.1f}%")
                    
                    # Visual bar chart
                    st.progress(probs[1])
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

with tab2:
    st.header("Video Deepfake Detection")
    
    uploaded_video = st.file_uploader(
        "Upload a video (MP4, AVI, MOV)",
        type=['mp4', 'avi', 'mov'],
        key='video_uploader'
    )
    
    if uploaded_video is not None:
        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        
        # Display video
        st.video(video_path)
        
        # Process video
        frame_skip = st.slider("Frame Skip (for faster processing)", 1, 30, 5)
        
        if st.button("🔍 Analyze Video", type="primary"):
            with st.spinner("🎬 Processing video frames..."):
                try:
                    model = load_model()
                    predictions = process_video(model, video_path, frame_skip)
                    
                    if predictions:
                        # Calculate statistics
                        fake_count = sum(1 for p in predictions if p['prediction'] == 0)
                        real_count = sum(1 for p in predictions if p['prediction'] == 1)
                        total_frames = len(predictions)
                        
                        fake_percentage = (fake_count / total_frames) * 100
                        avg_confidence = np.mean([p['confidence'] for p in predictions])
                        
                        # Display results
                        st.success("✅ Video analysis complete!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Frames Analyzed", total_frames)
                        with col2:
                            st.metric("Fake Frames", fake_count)
                        with col3:
                            st.metric("Real Frames", real_count)
                        with col4:
                            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                        
                        # Final verdict
                        st.write("")
                        if fake_percentage > 50:
                            st.error(f"⚠️ **VERDICT: FAKE** ({fake_percentage:.1f}% of frames detected as fake)")
                        else:
                            st.success(f"✅ **VERDICT: REAL** ({100-fake_percentage:.1f}% of frames detected as real)")
                        
                        # Frame-by-frame results
                        with st.expander("📊 View Frame-by-Frame Analysis"):
                            import pandas as pd
                            df = pd.DataFrame([
                                {
                                    'Frame': p['frame'],
                                    'Prediction': 'FAKE' if p['prediction'] == 0 else 'REAL',
                                    'Confidence': f"{p['confidence']:.2f}%",
                                    'Fake Prob': f"{p['probabilities'][0]*100:.1f}%",
                                    'Real Prob': f"{p['probabilities'][1]*100:.1f}%"
                                }
                                for p in predictions
                            ])
                            st.dataframe(df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                finally:
                    # Clean up temp file
                    Path(video_path).unlink(missing_ok=True)

with tab3:
    st.header("Model Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Training Performance")
        st.write("""
        - **Training Accuracy**: 99.24%
        - **Validation Accuracy**: 95.04%
        - **Test Accuracy**: 94.00%
        - **Training Time**: 4h 11m
        """)
        
        st.subheader("🔧 Model Architecture")
        st.write("""
        - **Base Model**: EfficientNet-B0
        - **Parameters**: 4,010,110
        - **Model Size**: 15.3 MB
        - **Input Size**: 224×224
        """)
    
    with col2:
        st.subheader("📈 Dataset Information")
        st.write("""
        - **Training Set**: 11,633 images
        - **Validation Set**: 2,400 images
        - **Test Set**: 2,400 images
        - **Total Images**: 16,433
        - **Classes**: Real / Fake
        """)
        
        st.subheader("💻 Hardware Used")
        st.write("""
        - **GPU**: NVIDIA GTX 1650 (4GB)
        - **Training Epochs**: 30
        - **Batch Size**: 16
        - **Optimizer**: Adam (lr=0.0001)
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🔍 Lightweight Deepfake Detection System</p>
    <p>Built with PyTorch & Streamlit | Model: EfficientNet-B0</p>
</div>
""", unsafe_allow_html=True)
