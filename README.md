# Real-Time Facial Recognition with LBPH

A lightweight real-time facial recognition project built with Python and OpenCV using the LBPH (Local Binary Patterns Histograms) algorithm.

This repository implements an end-to-end workflow for:
- collecting frames from a webcam
- preparing training images
- training an LBPH recognizer
- running single-image tests
- performing live webcam inference

Compared with the original repository README, this version documents the actual project files, clarifies the workflow, and summarizes the experimental results reported in the accompanying write-up.

## Project Summary

This project explores classical computer-vision-based face recognition using Haar cascade face detection and LBPH face recognition. The pipeline collects facial images, preprocesses them, trains an LBPH model, and then performs real-time recognition against a stored client image dataset.

In the accompanying project write-up, the evaluation reports:
- about 97% mean accuracy with a 36-photo dataset for one subject
- about 51% mean accuracy with a 19-photo dataset for another subject
- 50-frame trials per confidence setting
- an effective cross-dataset confidence threshold around 40-50
- about 5 seconds end-to-end runtime in the recorded setup

The methodology also notes that lower LBPH confidence values indicate a better match, and that a confidence threshold can be used to decide whether a face should be accepted as recognized.

## Repository Structure

The current repository contains these main files:
- `faceRecognition.py` - shared helper functions for detection, training, drawing, and labeling
- `tester.py` - trains the recognizer and runs recognition on a single test image
- `videoTester.py` - runs real-time webcam recognition using a saved LBPH model
- `videotoimg.py` - captures webcam frames and saves them as images
- `resizeImages.py` - resizes the training dataset images
- `trainingData.yml` - saved OpenCV LBPH model
- sample frame images (`frame7.jpg`, `frame15.jpg`, etc.)
- the original `Readme.md` file, which currently contains only a short setup guide

## How the Pipeline Works

### 1. Face Detection
The project uses a Haar cascade classifier to detect face bounding boxes in each frame or image.

### 2. Preprocessing
Input images are converted to grayscale before detection and recognition. The supporting write-up also highlights preprocessing steps such as resizing, grayscale conversion, and histogram equalization to improve consistency and performance.

### 3. Feature Extraction with LBPH
LBPH transforms local pixel neighborhoods into binary patterns, then builds region-wise histograms that represent facial texture. Those local histograms are concatenated into one feature vector for recognition.

### 4. Training
Training images are grouped by person ID. Each folder name acts as the integer label used by the recognizer.

### 5. Recognition
At inference time, the model predicts:
- a label (the closest stored identity)
- a confidence value (distance-based score)

Lower confidence means a better match, so recognition should only be accepted when confidence is below a chosen threshold.

## Requirements

- Python 3.9+
- `opencv-contrib-python`
- `numpy`

Install dependencies:

```bash
pip install opencv-contrib-python numpy
```

`opencv-contrib-python` is required because OpenCV's LBPH face recognizer lives in the `cv2.face` module.

## Recommended Folder Layout

```text
Real-Time-Facial-Recognition-with-LBPH-Algorithm/
├── faceRecognition.py
├── tester.py
├── videoTester.py
├── videotoimg.py
├── resizeImages.py
├── trainingData.yml
├── trainingImages/
│   ├── 0/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── 1/
│       ├── img1.jpg
│       └── img2.jpg
├── TestImages/
│   └── Tank.jpg
└── HaarCascade/
    └── haarcascade_frontalface_default.xml
```

Notes:
- each subfolder in `trainingImages/` should be named with an integer label such as `0`, `1`, `2`, etc.
- those numeric labels must match the name mapping used in `tester.py` and `videoTester.py`

## Usage

### Capture sample images from webcam

```bash
python videotoimg.py
```

This script captures webcam frames and stores them as JPG files.

### Resize collected images

```bash
python resizeImages.py
```

Use this step if you want a uniform training set size before training.

### Train the model and test on one image

```bash
python tester.py
```

This script:
1. loads training images from `trainingImages/`
2. trains an LBPH recognizer
3. writes the trained model to `trainingData.yml`
4. loads a test image from `TestImages/`
5. predicts the identity for each detected face

### Run real-time webcam recognition

```bash
python videoTester.py
```

This script loads the saved `trainingData.yml` model and performs recognition on frames captured from your webcam.

## Current Limitations in the Original Code

After reviewing the current repository, these are the main issues worth addressing:

1. **The cascade classifier is reloaded on every detection call**
   - In the current `faceRecognition.py`, the cascade is constructed inside `faceDetection`, which adds avoidable overhead on every frame.

2. **Face ROI slicing has width/height bugs**
   - Several slices use `y:y+w` or `x:x+h` instead of `y:y+h` and `x:x+w`, which can crop the wrong region and reduce recognition accuracy.

3. **`videoTester.py` performs redundant drawing and display work**
   - The current loop draws rectangles once, displays the frame, then loops again to predict, redraw, and display again.

4. **The confidence logic in live recognition is inconsistent**
   - The write-up says lower confidence is better, but the current `videoTester.py` labels a face as recognized when confidence is greater than the threshold.

5. **The training/inference flow is tightly coupled**
   - `tester.py` retrains the model every run by default, even when a saved `trainingData.yml` already exists.

6. **The source files are stored as one-line scripts**
   - This makes maintenance and debugging much harder than necessary.

## Performance Improvements Applied in the Updated Version

The improved version I prepared focuses on practical speed and correctness gains:

- cache the Haar cascade instead of rebuilding it for every frame
- apply a reusable grayscale + CLAHE preprocessing path
- fix all ROI slicing bugs (`x:x+w`, `y:y+h`)
- separate model loading from retraining when possible
- remove duplicate `imshow` / `waitKey` calls from the webcam loop
- save video frames at a configurable interval instead of every single frame
- add basic argument parsing and safer error handling
- make the code easier to extend and maintain

These changes should improve real-time responsiveness and make the results more stable under varying lighting.

## Research Context

This repository is also a portfolio-quality research project. Based on the materials you shared, it represents an end-to-end computer vision workflow: face detection, preprocessing, LBPH training, webcam inference, and threshold-based evaluation.

Your public Google Scholar entry for the provided scholar ID currently presents this work in the context of Computer Vision and Machine Learning, which fits well with positioning this repository as both:
- a practical OpenCV face recognition implementation
- a small research prototype focused on performance tradeoff analysis

## Future Improvements

If you want to push this further, the next best upgrades would be:
- detect faces on a downscaled frame and map boxes back to the original image
- add a configurable JSON file for label-to-name mappings
- automatically skip blurry or duplicate captured frames
- evaluate precision/recall or false-accept / false-reject tradeoffs
- compare LBPH against a stronger embedding-based recognizer
- add optional dataset balancing and augmentation
- package training and inference into a cleaner CLI interface

## Author

**Qinyang Tan**

This project reflects strong hands-on work in classical computer vision, applied machine learning, and reproducible experimentation. It is a good showcase of practical ML engineering: building a full pipeline, measuring performance, and identifying threshold tradeoffs instead of treating the model as a black box.

## License

MIT License
