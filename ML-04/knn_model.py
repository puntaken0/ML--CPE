from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def train_knn(X_train, y_train, n_neighbors=5):
    # Pipeline สำหรับ Scale ข้อมูลและเทรนโมเดล KNN
    scaler = Pipeline([
        ("scaler", StandardScaler()),
    ])

    X_train_scaled = scaler.fit_transform(X_train)

    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean")
    model.fit(X_train_scaled, y_train)

    return model, scaler

def predict_knn(model, scaler, X_test):
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    return predictions