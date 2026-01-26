"""
Simple CLI Inference Script for Deepfake Detection
Compatible with EfficientNet-B0 model
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Model Definition
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path):
    """Load the trained model"""
    model = LightweightDeepfakeDetector()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict_image(model, image_path):
    """Predict if an image is real or fake"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()

def predict_video(model, video_path, frame_skip=5):
    """Predict on video frames"""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None
    
    frame_count = 0
    predictions = []
    
    # Get total frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = total_frames // frame_skip
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, Processing every {frame_skip} frames")
    
    pbar = tqdm(total=frames_to_process, desc="Processing frames")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for faster processing
        if frame_count % frame_skip == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            img_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
            
            # Predict
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item() * 100
            
            predictions.append({
                'frame': frame_count,
                'prediction': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy()
            })
            
            pbar.update(1)
        
        frame_count += 1
    
    pbar.close()
    cap.release()
    
    return predictions

def print_results(file_path, pred_class, confidence, probs):
    """Print prediction results"""
    print("\n" + "="*60)
    print(f"File: {file_path}")
    print("="*60)
    
    if pred_class == 0:
        print("⚠️  PREDICTION: FAKE")
        print("   This content appears to be AI-generated or manipulated.")
    else:
        print("✅ PREDICTION: REAL")
        print("   This content appears to be authentic.")
    
    print(f"\nConfidence: {confidence:.2f}%")
    print(f"\nProbability Breakdown:")
    print(f"  • Fake: {probs[0]*100:.2f}%")
    print(f"  • Real: {probs[1]*100:.2f}%")
    print("="*60 + "\n")

def print_video_results(video_path, predictions):
    """Print video analysis results"""
    if not predictions:
        print("No predictions available")
        return
    
    # Calculate statistics
    fake_count = sum(1 for p in predictions if p['prediction'] == 0)
    real_count = sum(1 for p in predictions if p['prediction'] == 1)
    total_frames = len(predictions)
    
    fake_percentage = (fake_count / total_frames) * 100
    real_percentage = (real_count / total_frames) * 100
    avg_confidence = np.mean([p['confidence'] for p in predictions])
    
    print("\n" + "="*60)
    print(f"Video: {video_path}")
    print("="*60)
    
    print(f"\n📊 Analysis Results:")
    print(f"  • Frames Analyzed: {total_frames}")
    print(f"  • Fake Frames: {fake_count} ({fake_percentage:.1f}%)")
    print(f"  • Real Frames: {real_count} ({real_percentage:.1f}%)")
    print(f"  • Average Confidence: {avg_confidence:.2f}%")
    
    print(f"\n🎯 Final Verdict:")
    if fake_percentage > 50:
        print(f"  ⚠️  VIDEO IS LIKELY FAKE")
        print(f"     {fake_percentage:.1f}% of analyzed frames detected as manipulated")
    else:
        print(f"  ✅ VIDEO IS LIKELY REAL")
        print(f"     {real_percentage:.1f}% of analyzed frames detected as authentic")
    
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection - CLI Inference Tool")
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                        help='Path to model file (default: models/best_model.pth)')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--frame-skip', type=int, default=5,
                        help='Process every Nth frame in video (default: 5)')
    parser.add_argument('--batch', type=str, help='Path to directory with images/videos to process')
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Load model
    print(f"Loading model from {model_path}...")
    print(f"Using device: {DEVICE}")
    model = load_model(model_path)
    print("Model loaded successfully!\n")
    
    # Process image
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            return
        
        print(f"Processing image: {image_path}")
        pred_class, confidence, probs = predict_image(model, image_path)
        print_results(image_path, pred_class, confidence, probs)
    
    # Process video
    elif args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            return
        
        predictions = predict_video(model, video_path, args.frame_skip)
        if predictions:
            print_video_results(video_path, predictions)
    
    # Process batch
    elif args.batch:
        batch_dir = Path(args.batch)
        if not batch_dir.exists():
            print(f"Error: Directory not found: {batch_dir}")
            return
        
        # Find all images and videos
        image_extensions = ['.jpg', '.jpeg', '.png']
        video_extensions = ['.mp4', '.avi', '.mov']
        
        files = []
        for ext in image_extensions + video_extensions:
            files.extend(batch_dir.glob(f'*{ext}'))
            files.extend(batch_dir.glob(f'*{ext.upper()}'))
        
        if not files:
            print(f"No image or video files found in {batch_dir}")
            return
        
        print(f"Found {len(files)} files to process\n")
        
        for file_path in files:
            ext = file_path.suffix.lower()
            
            if ext in image_extensions:
                pred_class, confidence, probs = predict_image(model, file_path)
                print_results(file_path, pred_class, confidence, probs)
            
            elif ext in video_extensions:
                predictions = predict_video(model, file_path, args.frame_skip)
                if predictions:
                    print_video_results(file_path, predictions)
    
    else:
        print("Error: Please specify --image, --video, or --batch")
        parser.print_help()

if __name__ == '__main__':
    main()
