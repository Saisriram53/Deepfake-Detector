# 🚀 Quick Start Guide

## Prerequisites

Make sure you have:
- Python 3.11 installed
- Virtual environment activated (`venv311`)
- All dependencies installed (run `pip install -r requirements.txt`)
- Trained model at `models/best_model.pth`

## 1. Web Interface (Streamlit)

### Start the App

```bash
# Activate virtual environment
.\venv311\Scripts\Activate.ps1

# Run Streamlit app
streamlit run app_simple.py
```

The app will open in your browser at `http://localhost:8501`

### Features
- 📸 **Image Detection**: Upload single images (JPG, PNG)
- 🎥 **Video Detection**: Upload videos (MP4, AVI, MOV)
- 📊 **Statistics**: View model performance metrics
- Real-time predictions with confidence scores

---

## 2. Command Line Interface (CLI)

### Single Image

```bash
python inference_simple.py --image path/to/image.jpg
```

### Single Video

```bash
python inference_simple.py --video path/to/video.mp4
```

### Batch Processing

Process all images/videos in a directory:

```bash
python inference_simple.py --batch path/to/folder
```

### Options

- `--model`: Path to model file (default: `models/best_model.pth`)
- `--frame-skip`: Process every Nth frame in video (default: 5)

---

## Examples

### Example 1: Analyze an image
```bash
python inference_simple.py --image data/1000_videos/test/fake/frame_001.jpg
```

### Example 2: Analyze a video (process every 10th frame)
```bash
python inference_simple.py --video sample_video.mp4 --frame-skip 10
```

### Example 3: Batch process test images
```bash
python inference_simple.py --batch data/1000_videos/test/fake
```

---

## Understanding Results

### Image/Single Frame Output
```
==============================================================
File: sample.jpg
==============================================================
✅ PREDICTION: REAL
   This content appears to be authentic.

Confidence: 98.45%

Probability Breakdown:
  • Fake: 1.55%
  • Real: 98.45%
==============================================================
```

### Video Output
```
==============================================================
Video: sample.mp4
==============================================================

📊 Analysis Results:
  • Frames Analyzed: 60
  • Fake Frames: 52 (86.7%)
  • Real Frames: 8 (13.3%)
  • Average Confidence: 92.34%

🎯 Final Verdict:
  ⚠️  VIDEO IS LIKELY FAKE
     86.7% of analyzed frames detected as manipulated
==============================================================
```

---

## Troubleshooting

### Model Not Found
```bash
Error: Model file not found: models/best_model.pth
```
**Solution**: Make sure you've trained the model or download the pre-trained weights.

### CUDA Out of Memory
If you get CUDA out of memory errors with videos:
```bash
python inference_simple.py --video sample.mp4 --frame-skip 30
```
Increase `--frame-skip` to process fewer frames.

### Dependencies Missing
```bash
pip install -r requirements.txt
```

---

## Performance Tips

1. **For Videos**: 
   - Use `--frame-skip 10` or higher for faster processing
   - Smaller videos = faster results

2. **For Batch Processing**:
   - Process in smaller batches if you have memory constraints
   - Use GPU for faster inference (automatically detected)

3. **GPU vs CPU**:
   - GPU: ~50-100 images/sec
   - CPU: ~5-10 images/sec

---

## Model Information

- **Architecture**: EfficientNet-B0
- **Parameters**: 4,010,110
- **Model Size**: 15.3 MB
- **Test Accuracy**: 94%
- **Input Size**: 224×224 RGB

---

## Next Steps

1. Try the web interface: `streamlit run app_simple.py`
2. Test with your own images/videos
3. Check the full documentation in `README.md`

---

**Need Help?** Check the README.md or open an issue on GitHub.
