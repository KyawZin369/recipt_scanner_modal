"""
Train a scanner classifier: "has text" vs "no text" (runs without TensorFlow for macOS compatibility).
Saves the model to models/scanner_sklearn.joblib for use in scanner.py.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

IMG_SIZE = (128, 128)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "scanner_sklearn.joblib"
DATA_DIR = Path(__file__).resolve().parent / "data"


def generate_synthetic_data(num_samples=400, img_size=IMG_SIZE):
    """Generate simple synthetic images for demo training (no real data needed)."""
    with_text = []
    no_text = []
    rng = np.random.default_rng(42)

    for _ in range(num_samples // 2):
        img = rng.integers(0, 256, (*img_size, 3), dtype=np.uint8)
        no_text.append(img)
        img = rng.integers(100, 200, (*img_size, 3), dtype=np.uint8)
        for i in range(0, img_size[1], 15):
            img[i : i + 4, :, :] = 20
        with_text.append(img)

    X = np.array(with_text + no_text, dtype=np.float32) / 255.0
    y = np.array([1] * (num_samples // 2) + [0] * (num_samples // 2), dtype=np.int32)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def load_data_from_folders():
    """Load images from data/with_text and data/no_text if they exist."""
    with_text_dir = DATA_DIR / "with_text"
    no_text_dir = DATA_DIR / "no_text"
    if not with_text_dir.is_dir() or not no_text_dir.is_dir():
        return None, None

    def load_folder(folder, label):
        X, y = [], []
        for path in folder.glob("*"):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            try:
                img = Image.open(path).convert("RGB")
                arr = np.array(img.resize(IMG_SIZE), dtype=np.float32) / 255.0
                X.append(arr)
                y.append(label)
            except Exception:
                continue
        return X, y

    X1, y1 = load_folder(with_text_dir, 1)
    X0, y0 = load_folder(no_text_dir, 0)
    if not X1 and not X0:
        return None, None
    X = np.array(X1 + X0)
    y = np.array(y1 + y0)
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


def main():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "with_text").mkdir(exist_ok=True)
    (DATA_DIR / "no_text").mkdir(exist_ok=True)

    X, y = load_data_from_folders()
    if X is None:
        print("No images in data/with_text and data/no_text. Using synthetic data for demo.")
        X, y = generate_synthetic_data()

    # Flatten for sklearn
    n = X.shape[0]
    X_flat = X.reshape(n, -1)

    X_train, X_val, y_train, y_val = train_test_split(X_flat, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    acc = accuracy_score(y_val, pred)
    print(f"Validation accuracy: {acc:.2%}")

    # Save with metadata so scanner knows input shape
    joblib.dump({"model": model, "img_size": IMG_SIZE}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
