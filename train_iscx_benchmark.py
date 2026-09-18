"""
Trains a separate 5-class XGBoost benchmark on the ISCX-URL2016 dataset
(data/raw/All.csv). This dataset ships as pre-extracted lexical features
with no raw URL column, so it can't be merged into the main PhishGuard
pipeline (preprocess.py -> build_features.py -> train_model.py); it is
evaluated here as an independent benchmark instead.
"""
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


def train_iscx_benchmark():
    print("Loading ISCX-URL2016 (All.csv)...")
    df = pd.read_csv('data/raw/All.csv')
    print(f"Dataset loaded. Total rows: {len(df)}")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['URL_Type_obf_Type'])
    X = df.drop(columns=['URL_Type_obf_Type'])
    X = X.fillna(-1.0).replace([float('inf'), float('-inf')], -1.0)

    print(f"Classes: {list(label_encoder.classes_)}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training XGBoost (multi-class)...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)

    print("\n --- ISCX Benchmark Evaluation ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': model, 'label_encoder': label_encoder}, 'models/iscx_benchmark_xgb.pkl')
    print("\nModel saved to 'models/iscx_benchmark_xgb.pkl'")


if __name__ == '__main__':
    train_iscx_benchmark()
