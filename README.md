
# Deepfake Detection System 🔍

A **lightweight, efficient** deepfake detection system using **EfficientNet-B0** architecture.  
Upload images or videos and get instant, accurate predictions with confidence scores.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.53+-FF4B4B.svg)](https://streamlit.io/)

## 🌟 Features

- 📸 **Image & Video Analysis:** Supports JPG, PNG images and MP4, AVI, MOV videos
- 🧠 **Efficient Architecture:** EfficientNet-B0 - lightweight yet powerful
- 🎯 **High Accuracy:** 94% test accuracy, 95% validation accuracy
- ⚡ **Fast Inference:** Optimized for GTX 1650 4GB GPU
- 📊 **Interactive UI:** Beautiful Streamlit interface with confidence scores
- 💻 **CLI Tool:** Batch processing capability via command line
- 🔧 **Memory Optimized:** Runs smoothly on consumer-grade hardware

## 🏗️ Architecture

### Model: EfficientNet-B0

```
Input Image (224×224×3)
    ↓
[EfficientNet-B0 Backbone]
    ↓ (Pretrained on ImageNet)
[Feature Extraction] (1280 features)
    ↓
[Dropout 0.2]
    ↓
[Fully Connected Layer]
    ↓
Output: Real (1) vs Fake (0)
```

**Key Components:**
- **Backbone:** EfficientNet-B0 (pretrained on ImageNet)
- **Input Size:** 224×224 RGB images
- **Parameters:** 4,010,110 (4M)
- **Model Size:** 15.3 MB
- **Training:** 30 epochs on 16,433 images

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- CUDA-capable GPU (recommended) or CPU
- 4GB+ GPU VRAM (for training)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/Saisriram53/Deepfake-Detector.git
cd Deepfake-Detector
```

2. **Create virtual environment**

```bash
python -m venv venv311
# On Windows:
.\venv311\Scripts\Activate.ps1
# On Linux/Mac:
source venv311/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download the trained model**

Place your trained `best_model.pth` (15.3 MB) in the `models/` folder, or train your own using `train_deepfake_optimized.ipynb`.

### Running the Application

**Web Interface (Streamlit):**
```bash
streamlit run app_simple.py
```

**Command Line Interface:**
```bash
# Single image
python inference_simple.py --image path/to/image.jpg

# Video analysis
python inference_simple.py --video path/to/video.mp4

# Batch processing
python inference_simple.py --batch path/to/folder
```

The web app will launch at [http://localhost:8501](http://localhost:8501)

## 📖 Usage

### Web Interface (Streamlit)

1. **Launch the app:** `streamlit run app_simple.py`
2. **Upload Image/Video:** Use the file uploader in the "Image Detection" or "Video Detection" tab
3. **View Results:** Get instant predictions with confidence scores
4. **Statistics:** Check the "Statistics" tab for model performance metrics

### Command Line Interface

```bash
# Analyze a single image
python inference_simple.py --image test.jpg

# Analyze a video (process every 5th frame)
python inference_simple.py --video test.mp4 --frame-skip 5

# Batch process a folder
python inference_simple.py --batch data/test/fake --frame-skip 10
```

**CLI Options:**
- `--model`: Path to model file (default: `models/best_model.pth`)
- `--frame-skip`: Process every Nth frame in videos (default: 5)
- `--batch`: Process all images/videos in a directory

## 📂 Project Structure

```
Deepfake-Detector/
├── app_simple.py               # Streamlit web application
├── inference_simple.py         # CLI inference tool
├── train_deepfake_optimized.ipynb  # Training notebook
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── QUICKSTART.md              # Quick start guide
│
├── .streamlit/
│   └── config.toml            # Streamlit configuration
│
├── .devcontainer/             # VS Code dev container config
├── .github/                   # GitHub Actions workflows
│
├── assets/                    # UI assets and samples
├── models/                    # Saved model (best_model.pth)
│   └── best_model.pth        # Trained EfficientNet-B0 (15.3 MB)
│
└── data/                      # Dataset (not in git)
    └── 1000_videos/
        ├── train/
        ├── validation/
        └── test/
```

## 🎓 Training Your Own Model

### Dataset Preparation

Organize your dataset in this structure:

```
data/1000_videos/
├── train/
│   ├── fake/    # Fake/manipulated images
│   └── real/    # Real/authentic images
├── validation/
│   ├── fake/
│   └── real/
└── test/
    ├── fake/
    └── real/
```

### Training

Open and run `train_deepfake_optimized.ipynb` in Jupyter:

```bash
jupyter notebook train_deepfake_optimized.ipynb
```

**Training Configuration:**
- **Epochs:** 30
- **Batch Size:** 16 (optimized for 4GB GPU)
- **Learning Rate:** 0.0001
- **Optimizer:** Adam
- **Image Size:** 224×224
- **Augmentation:** Random flip, rotation, color jitter

**Hardware Requirements:**
- **GPU:** NVIDIA GTX 1650 4GB or better
- **Training Time:** ~4 hours on GTX 1650

The notebook will:
1. Load and preprocess the dataset
2. Train the EfficientNet-B0 model
3. Save the best model to `models/best_model.pth`
4. Generate training curves to `models/training_history.png`
5. Evaluate on test set

### Expected Results

After training, you should achieve:
- **Training Accuracy:** ~99%
- **Validation Accuracy:** ~95%
- **Test Accuracy:** ~94%

## 📊 Performance Benchmarks

### Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 94.00% |
| **Validation Accuracy** | 95.04% |
| **Training Accuracy** | 99.24% |
| **Model Size** | 15.3 MB |
| **Parameters** | 4,010,110 |
| **Training Time** | 4h 11m (GTX 1650) |

### Dataset Statistics

| Split | Images | Fake | Real |
|-------|--------|------|------|
| **Training** | 11,633 | ~50% | ~50% |
| **Validation** | 2,400 | ~50% | ~50% |
| **Test** | 2,400 | ~50% | ~50% |
| **Total** | 16,433 | - | - |

### Inference Speed

| Hardware | Images/sec | Video FPS |
|----------|------------|-----------|
| **GTX 1650 (4GB)** | ~50-100 | 30-60 fps |
| **CPU (Intel i5)** | ~5-10 | 5-10 fps |
|---------|----------|-----------|--------|----------|
| FaceForensics++ | 96.8% | 95.2% | 97.1% | 96.1% |
| Celeb-DF | 94.3% | 92.8% | 94.9% | 93.8% |
| DFDC | 91.7% | 90.2% | 92.3% | 91.2% |

### Speed

| Hardware | FPS | Batch Size |
|----------|-----|------------|
| RTX 3080 | 18.5 | 8 |
## 🛠️ Technical Details

### Model Architecture

```python
class LightweightDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # EfficientNet-B0 backbone (pretrained on ImageNet)
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Binary classification head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 2)  # 2 classes: fake (0) and real (1)
        )
```

### Training Details

- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.0001)
- **LR Scheduler:** ReduceLROnPlateau (patience=3, factor=0.5)
- **Data Augmentation:**
  - Random horizontal flip (p=0.5)
  - Random rotation (±10°)
  - Color jitter (brightness, contrast, saturation ±20%)
- **Normalization:** ImageNet mean/std

## 🐛 Troubleshooting

### CUDA Out of Memory

If training fails with OOM errors:
- Reduce batch size in notebook: `BATCH_SIZE = 8`
- Use `--frame-skip` for videos: `python inference_simple.py --video test.mp4 --frame-skip 10`

### Model Not Found

```bash
# Make sure model exists
ls models/best_model.pth

# Or download from releases
# https://github.com/Saisriram53/Deepfake-Detector/releases
```

### Dependencies Issues

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check versions
python -c "import torch; print(torch.__version__)"
python -c "import streamlit; print(streamlit.__version__)"
```

## 📝 Citation

If you use this code in your research or project, please cite:

```bibtex
@misc{deepfake-detector-2026,
  title={Lightweight EfficientNet-B0 Deepfake Detection System},
  author={Saisriram53},
  year={2026},
  publisher={GitHub},
  url={https://github.com/Saisriram53/Deepfake-Detector}
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [EfficientNet](https://arxiv.org/abs/1905.11946) - Efficient Convolutional Neural Networks
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Streamlit](https://streamlit.io/) - Web application framework
- Dataset: 1000 Deepfake Videos Dataset

## 📧 Contact

**GitHub:** [@Saisriram53](https://github.com/Saisriram53)  
**Repository:** [Deepfake-Detector](https://github.com/Saisriram53/Deepfake-Detector)

For questions or issues, please open an issue on GitHub.

---

<div align="center">
  <p>⭐ Star this repo if you find it helpful!</p>
  <p>Made with ❤️ for accurate deepfake detection</p>
</div>

---

**Built with ❤️ using PyTorch, EfficientViT, and Streamlit**

