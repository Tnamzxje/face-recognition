import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle

# Đọc dữ liệu
print("Đang đọc dữ liệu...")
data = pd.read_csv('encoded-images-data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Load LabelEncoder đã có
print("Đang load LabelEncoder từ file...")
with open('labels.pkl', 'rb') as f:
    le = pickle.load(f)

# Chuyển đổi nhãn
print("Đang chuyển đổi nhãn...")
y = le.transform(y)

# Lọc ra các lớp có ít nhất 2 mẫu
unique_classes, class_counts = np.unique(y, return_counts=True)
valid_classes = unique_classes[class_counts >= 2]
mask = np.isin(y, valid_classes)
X = X[mask]
y = y[mask]

print(f"Số lớp sau khi lọc: {len(valid_classes)}")
print(f"Số mẫu sau khi lọc: {len(X)}")

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM
print("Training SVM with linear kernel...")
clf = SVC(C=1, kernel='linear', probability=True, random_state=42)
clf.fit(X_train, y_train)

# Đánh giá
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Lưu model và label encoder
print("\nSaving SVM classifier to svm_classifier.pkl...")
with open('svm_classifier.pkl', 'wb') as f:
    pickle.dump((le, clf), f)
print("Done!")
