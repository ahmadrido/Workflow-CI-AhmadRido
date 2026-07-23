import pandas as pd
import mlflow
import mlflow.sklearn
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Hapus folder model lama jika sudah ada agar tidak bentrok saat dioverwrite
if os.path.exists("model_dir"):
    shutil.rmtree("model_dir")

def train_model():
    print("Memuat dataset untuk CI/CD Pipeline...")
    df = pd.read_csv('obesity_preprocessing/obesity_ready.csv')
    
    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Menggunakan parameter terbaik dari hasil Hyperparameter Tuning sebelumnya
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=2, random_state=42)
    
    print("Melatih model RandomForest...")
    rf.fit(X_train, y_train)

    # Simpan model secara lokal ke folder "model_dir" agar mudah di-build oleh Docker
    mlflow.sklearn.save_model(rf, "model_dir")
    print("Model berhasil disimpan ke direktori 'model_dir' siap untuk di-build Docker-nya.")

if __name__ == "__main__":
    train_model()