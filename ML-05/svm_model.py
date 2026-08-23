from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def train_svm(X_train, y_train, kernel="rbf", pca_components=150):
    # Pipeline: StandardScaler + PCA เพื่อลดมิติข้อมูลจาก 10,000 เหลือ 150 มิติ
    scaler = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(pca_components, *X_train.shape),
                    whiten=True, random_state=42)),
    ])

    X_train_scaled = scaler.fit_transform(X_train)

    # สร้างโมเดล SVM ตาม Kernel ที่กำหนด
    if kernel == "linear":
        model = SVC(kernel="linear", C=1.0, cache_size=1000, random_state=42)
    elif kernel == "poly":
        model = SVC(kernel="poly", degree=3, C=10, cache_size=1000, random_state=42)
    else:  # rbf
        model = SVC(kernel="rbf", C=10, gamma="scale", cache_size=1000, random_state=42)

    model.fit(X_train_scaled, y_train)
    return model, scaler

def predict_svm(model, scaler, X_test):
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    return predictions