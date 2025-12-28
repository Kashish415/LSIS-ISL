# LSIS-ISL

Real-time Indian Sign Language (ISL) recognition using MediaPipe and TensorFlow. This system can recognize 35 ISL signs (A-Z alphabets and 1-9 numbers) through webcam with 99% accuracy.

## Setup

Step 1: Download Dataset

Go to Kaggle and download the ISL dataset:

* Dataset Link: https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl

Step 2: Upload Dataset to Google Colab

1. Open your Colab notebook.
2. In Colab, click the Files icon on the left sidebar
3. Create a new folder: Right-click → New folder → Name it isl_dataset
4. Upload the Indian folder inside isl_dataset.
   OR
Alternative (Faster): Upload the zip file and extract it in Colab using the below command:

!unzip /content/archive.zip -d /content/isl_dataset/

Step 3: Train Your Own Model ( refer model_training.py file)

Step 4: After training completes, downloading the following files and save them to the vs code project folder:
- `isl_hand_model.h5`
- `label_encoder.pkl`
- `hand_landmarker.task`

Place these files in the project root directory

### Step 5: Create main.py ( vs code setup -> pip install -r requirements.txt)

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

## How It Works

1. **Hand Detection**: MediaPipe extracts 21 hand landmarks (x, y, z coordinates)
2. **Feature Extraction**: 63 features (21 landmarks × 3 coordinates) are extracted
3. **Classification**: Neural network predicts the sign
4. **Stabilization**: Predictions are stabilized over multiple frames to reduce jitter
5. **Display**: Result is shown on the video feed

## Technical Details

### Hand Landmark Extraction

MediaPipe Hand Landmarker detects 21 keypoints
Each keypoint has (x, y, z) coordinates
Total features: 21 × 3 = 63

### Neural Network Architecture

Input Layer:    63 features
Dense Layer 1:  256 neurons (ReLU) + BatchNorm + Dropout(0.4)
Dense Layer 2:  128 neurons (ReLU) + BatchNorm + Dropout(0.4)
Dense Layer 3:  64 neurons (ReLU) + Dropout(0.3)
Output Layer:   35 neurons (Softmax)

### Model Training
- Dataset: 100 images per class (3500 total)
- Train/Test Split: 80/20
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Callbacks: Early Stopping, ReduceLROnPlateau

### Model Pipeline 

Webcam Frame → RGB Conversion → MediaPipe Detection → 
Landmark Extraction → Feature Vector (63) → 
Neural Network → Softmax Prediction → Display Result

### Class Distribution

35 classes total:

* Alphabets: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
* Numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9

### Training Details

Dataset Size: 3,500 images (100 per class)
Train/Test Split: 80/20
Batch Size: 32
Epochs: 50 (with early stopping)
Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
