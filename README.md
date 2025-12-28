# LSIS-ISL

Real-time Indian Sign Language (ISL) recognition using MediaPipe and TensorFlow. This system can recognize 35 ISL signs (A-Z alphabets and 1-9 numbers) through webcam with 99%+ accuracy.

## Features

- Real-time hand landmark detection using MediaPipe
- 35 ISL signs recognition (A-Z, 1-9)
- 99.85% test accuracy
- Smooth prediction stabilization
- Easy-to-use webcam interface

## Dataset

Indian Sign Language Dataset: [Kaggle Link](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)

## Project Structure
```
lsis-isl-recognition/
├── train_model.ipynb          # Google Colab notebook for training
├── main.py                    # Real-time webcam recognition script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── models/                    # Trained model files (download separately)
    ├── isl_hand_model.h5
    ├── label_encoder.pkl
    └── hand_landmarker.task
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- 4GB RAM minimum

### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/lsis-isl-recognition.git
cd lsis-isl-recognition
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Trained Models

Download the pre-trained models from [Google Drive Link] or train your own using the Colab notebook.

Place these files in the project root directory:
- `isl_hand_model.h5`
- `label_encoder.pkl`
- `hand_landmarker.task`

## Usage

### Training the Model

1. Open `train_model.ipynb` in Google Colab
2. Upload the ISL dataset
3. Run all cells sequentially
4. Download the trained model files

### Running Real-Time Recognition
```bash
python main.py
```

**Controls:**
- Show hand signs in front of webcam
- Press `q` to quit

## Model Performance
```
Test Accuracy: 99.85%
Train Accuracy: 99.93%
Test Loss: 0.0024
```

**Model Architecture:**
- Input: 63 features (21 hand landmarks × 3 coordinates)
- Dense layers: 256 → 128 → 64
- Dropout & Batch Normalization for regularization
- Output: 35 classes (Softmax activation)

## How It Works

1. **Hand Detection**: MediaPipe extracts 21 hand landmarks (x, y, z coordinates)
2. **Feature Extraction**: 63 features (21 landmarks × 3 coordinates) are extracted
3. **Classification**: Neural network predicts the sign
4. **Stabilization**: Predictions are stabilized over multiple frames to reduce jitter
5. **Display**: Result is shown on the video feed

## Technical Details

### Hand Landmark Extraction
- MediaPipe Hand Landmarker model
- 21 keypoints per hand
- Normalized coordinates (0-1 range)

### Model Training
- Dataset: 100 images per class (3500 total)
- Train/Test Split: 80/20
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Callbacks: Early Stopping, ReduceLROnPlateau

### Real-Time Optimization
- Confidence threshold: 70%
- Stability frames: 3 consecutive predictions
- Mirror effect for natural interaction

## Project Context

This is part of the **Live Sign-Language Interpretation Service (LSIS)** project, aimed at removing communication barriers between deaf sign-language users and hearing individuals during online meetings.

### Future Enhancements
- [ ] Continuous sign recognition (sentences)
- [ ] Text-to-Speech integration
- [ ] Chrome extension development
- [ ] Speech-to-Sign avatar
- [ ] Multi-hand support
- [ ] ASL support

## Requirements
```
mediapipe==0.10.31
opencv-python==4.10.0.84
tensorflow==2.18.0
numpy==1.26.4
scikit-learn==1.5.2
```

## Troubleshooting

### Webcam not opening
```bash
# Test webcam
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Slow performance
- Close other applications
- Reduce frame processing rate
- Use GPU if available

### Import errors
```bash
pip install --upgrade mediapipe opencv-python tensorflow
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

- Dataset: [Kaggle ISL Dataset](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)
- MediaPipe: Google's ML framework
- TensorFlow: Model training framework

## Contact

For questions or suggestions, please open an issue or contact [your-email@example.com]

---

**Note**: Model files are not included in the repository due to size limitations. Please download them separately or train your own model using the provided Colab notebook.
