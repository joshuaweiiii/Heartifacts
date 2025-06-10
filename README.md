# **Heartifacts - ECG Beat Classification**
## **Project Description**

This project classifies ECG beats as **normal** or **heart attack related** using various machine learning models on the MIT-BIH Arrhythmia Database.

**Models Used:**
- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine

**Preprocessing Performed**
- Bandpass Filtering
- High Pass Filtering
- Normalization
- Feature Scaling
- Fourier Transform (Visualization Only)

## **Directory Structure**
```
Heartifacts/
│
├── finalDatabase/              # MIT-BIH Arrhythmia Database 
│
├── models/                     # Machine Learning models
│   ├── LogisticRegression.py
│   ├── RandomForest.py
│   ├── KNN.py
│   ├── SVM.py
│
├── pictures/                   # Visualizations
│   ├── BandPassFilterComparison.png
│   ├── HighPassFilterComparison.png
│   ├── FastFourierTransform.png
│   ├── QRS Complex.png
│   ├── FullRawFile.png
│   ├── 5SecExample.png
│
├── loadData.py                 # Loading ECG Data and generating main_df.csv
├── preprocessing.py            # Preprocessing pipeline → filtering + normalization
├── main_df.csv                 # Final processed dataset used by models
│
└── README.md                   # This file → project overview and structure

```

## **How to Run**
1. Run ``` preprocessing.py ``` to generate ``` main_df.csv ```
2. Run individual models in ``` models/ ``` folder:

```
python3 models/LogisticRegression.py
python3 models/RandomForest.py
python3 models/KNN.py
python3 models/SVM.py
```

## **Notes**
- The MIT-BIH Database is pre-downloaded into ``` finalDatabase/ ```.

- Large files (original main_df.csv) are not pushed to GitHub due to size limits.

- The project is structured for easy testing and comparison of ML models.

## **Future Improvements**
- Add 1D CNN model for comparison.

- Implement automated cross-validation for all models.

- Add training curve visualizations (accuracy vs iterations).

## **Credits**
Data: MIT-BIH Arrhythmia Database

Team: Darren Lam, Justin Yee, Joshua Wei
