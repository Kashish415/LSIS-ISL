# LSIS-ISL

Real-time Indian Sign Language (ISL) recognition using MediaPipe and TensorFlow. This system can recognize 35 ISL signs (A-Z alphabets and 1-9 numbers) through webcam with 99% accuracy.

## Dataset

Indian Sign Language Dataset: [Kaggle Link](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)

### Step 1: Clone Repository
```bash
git clone https://github.com/Kashish415/LSIS-ISL.git
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run from colab environment

Train your own model using the given Colab notebook.

Place these files in the project root directory, after downloading them and saving them to the vs code project folder:
- `isl_hand_model.h5`
- `label_encoder.pkl`
- `hand_landmarker.task`

### Step 4: Run main.py ( vs code setup)

Execute the main.py using terminal command:
```bash
python main.py
```

**Controls:**
- Show hand signs in front of the camera window
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
