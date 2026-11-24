# 1D-CNN for Mental Workload Classification (STEW EEG Dataset)

A 1D Convolutional Neural Network (TensorFlow/Keras) that classifies mental
cognitive workload into three levels (0, 1, 2) from raw multichannel EEG
signals in the **STEW (Simultaneous Task EEG Workload)** dataset.

## Dataset

- **Source:** Kaggle – *Mental Cognitive Workload EEG Data (STEW Dataset)*
- **Files used** (`.mat`):
  | File | Contents |
  |------|----------|
  | `dataset.mat` | Raw EEG, shape `(14, 19200, 45)` = channels × samples × subjects |
  | `rating.mat` | Subjective workload ratings, shape `(45, 1)` |
  | `class_012.mat` | Workload class labels (0, 1, 2), shape `(45, 1)` |
  | `three_class_one_hot.mat` | One-hot version of the labels (loaded, not used) |
- **Signal:** 14 EEG channels, 19,200 time samples per recording (~150 s at 128 Hz)

## Project Workflow

1. **Load data** with `scipy.io.loadmat`.
2. **Reshape** the EEG array to `(45, 19200, 14)` (samples, time, channels).
3. **Split** into train/test: 90% / 10% (`random_state=42`, stratified by class)
   → 40 training and 5 test recordings.
4. **Build the model** – 3 Conv1D blocks followed by a dense head:

   | Block | Layers |
   |-------|--------|
   | 1 | Conv1D(128, k=4, ReLU) → MaxPool(2) → BatchNorm → Dropout(0.5) |
   | 2 | Conv1D(64, k=8, ReLU) → MaxPool(2) → BatchNorm → Dropout(0.5) |
   | 3 | Conv1D(32, k=16, ReLU) → MaxPool(2) → BatchNorm → Dropout(0.5) |
   | Head | Flatten → Dense(100, ReLU) → Dense(output) |

5. **Train** with Adam, batch size 1, 2 epochs.
6. **Evaluate** with accuracy, confusion matrix, and a custom
   `multi_class_parameters()` function that reports accuracy,
   misclassification rate, sensitivity (recall), specificity, precision and F1.

## Requirements

```
python >= 3.10
numpy
pandas
scipy
scikit-learn
tensorflow >= 2.x (Keras 3)
```

Install with:

```bash
pip install numpy pandas scipy scikit-learn tensorflow
```

## How to Run

1. Download the STEW dataset from Kaggle and place the `.mat` files in a
   folder.
2. Update the paths in the data-loading cell (currently
   `../input/mental-cognitive-workload-eeg-data-stew-dataset/`).
3. Run the notebook top to bottom (`1d-cnn-stew-dataset.ipynb`).

## Current Results

| Metric | Value |
|--------|-------|
| Test accuracy | 0.40 (Keras) / 0.20 (confusion-matrix based) |
| Predictions | All test samples predicted as class 0 |

These results are **not meaningful** yet, because of the known issues below.

## Known Issues / Suggested Fixes

- **Output layer:** `Dense(1, activation='softmax')` always outputs 1.
  Use `Dense(3, activation='softmax')`.
- **Loss/label mismatch:** `categorical_crossentropy` expects one-hot labels,
  but integer labels are passed. Either use
  `sparse_categorical_crossentropy` with the integer labels, or pass
  `class012_onehot` as the targets.
- **Reshaping:** `dataset.resize(45, 19200, 14)` reinterprets memory and
  scrambles the data instead of reordering axes. Use
  `dataset = np.transpose(dataset, (2, 1, 0))`.
- **Tiny dataset:** only 45 recordings, with 5 in the test set, so metrics
  are very unstable. Consider cross-validation, or slicing each recording
  into shorter windows to create more training samples.
- **Training setup:** 2 epochs with batch size 1 is too little. Increase the
  epochs, use a larger batch size, and add a validation split.
- **Preprocessing:** normalize/standardize each channel and consider
  band-pass filtering the EEG.
- **Metric function:** `multi_class_parameters()` produces `nan` precision/F1
  when a class is never predicted.

## Future Work

- Fix the issues above and re-benchmark
- Windowed/segmented EEG input for data augmentation
- Compare with other models (LSTM, EEGNet, SVM on band-power features)
- Subject-independent evaluation

## Acknowledgements

- STEW dataset: Lim, W.L., Sourina, O., Wang, L.P. (2018),
  *STEW: Simultaneous Task EEG Workload Data Set*, IEEE TNSRE.
- Dataset hosted on Kaggle.

## License

Specify your license here (e.g., MIT).
