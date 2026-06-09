import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


X_train = np.load("X_train2.npy")
y_train = np.load("y_train2.npy")
X_test = np.load("X_test2.npy")
y_test = np.load("y_test2.npy")

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

print(f"Ukuran Training: {X_train.shape}, Ukuran Testing: {X_test.shape}")


NAMA_FITUR = [
    'Age', 'Gender', 'Academic Year', 'Study Hours', 'Exam Pressure',
    'Academic Performance', 'Anxiety Score', 'Depression Score', 'Sleep Hours',
    'Physical Activity', 'Social Support', 'Screen Time', 'Internet Usage',
    'Financial Stress', 'Family Expectation', 'Burnout Score'
]

# model_xgb = XGBClassifier(
#     n_estimators=100,
#     max_depth=6,
#     learning_rate=0.1,
#     objective='multi:softprob', 
#     random_state=42,
#     eval_metric='mlogloss'
# )

from lightgbm import LGBMClassifier

model_lgbm = LGBMClassifier(
    verbosity=-1,
    random_state=42,
    n_jobs=-1,
    class_weight = 'balanced',
    max_depth=10,
    learning_rate=0.05,
    n_estimators=500
)


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model_lgbm, X_train, y_train, cv=kfold, scoring='accuracy')

print(f"Akurasi tiap fold: {cv_scores}")
print(f" Rata-rata Akurasi CV: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")


model_lgbm.fit(X_train, y_train)

y_pred = model_lgbm.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Akhir pada Data Test: {test_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Moderate', 'High']))

print("\n MENGEKSTRAKSI FEATURE IMPORTANCE DARI XGBOOST")
importances = model_lgbm.feature_importances_

indeks_urut = np.argsort(importances)[::-1]

print("-" * 50)
print(" Rank | Nama Fitur Asli        | Bobot Pengaruh")
print("-" * 50)
for rank, idx in enumerate(indeks_urut[:5], 1):
    print(f"  {rank}   | {NAMA_FITUR[idx]:<22} | {importances[idx]:.4f}")
print("-" * 50)

MODEL_NAME = "model_stress3.pkl"
joblib.dump(model_lgbm, MODEL_NAME)

print(f"Model Berhasil disimpan dengan nama: '{MODEL_NAME}'")
