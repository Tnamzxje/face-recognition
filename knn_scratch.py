import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

# Hàm tính khoảng cách Euclidean
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Hàm KNN tự viết
def knn_predict(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_point in X_test:
        # Tính khoảng cách từ test_point đến tất cả điểm train
        distances = [euclidean_distance(test_point, x) for x in X_train]
        # Lấy k chỉ số nhỏ nhất
        k_indices = np.argsort(distances)[:k]
        k_labels = [y_train[i] for i in k_indices]
        # Lấy nhãn xuất hiện nhiều nhất
        most_common = Counter(k_labels).most_common(1)[0][0]
        y_pred.append(most_common)
    return np.array(y_pred)

if __name__ == '__main__':
    # Đọc dữ liệu
    data = pd.read_csv('encoded-images-data.csv')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Loại bỏ các lớp có ít hơn 2 mẫu
    label_counts = Counter(y)
    mask = np.array([label_counts[label] >= 2 for label in y])
    X = X[mask]
    y = y[mask]

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Dự đoán bằng KNN tự viết
    k = 3
    print(f"Chạy KNN tự viết với k={k}...")
    y_pred = knn_predict(X_train, y_train, X_test, k=k)

    # Đánh giá
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred)) 