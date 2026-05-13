from pathlib import Path

import joblib
import numpy as np
import skimage as ski
from skimage import exposure, filters
from skimage.feature import graycomatrix, graycoprops, hog
from skimage.restoration import denoise_tv_chambolle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CATEGORIES = ["healthy", "glioma", "meningioma", "pituitary", "tumor"]

DATASET_HEALTHY_DIR = Path("./dataset/healthy")
DATASET_TUMOR_DIR = Path("./dataset/tumor")
MODEL_OUTPUT_DIR = Path("./models")
IMAGE_SIZE = 256
RANDOM_STATE = 42


def list_image_files(folder: Path):
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and not path.name.startswith(".")
        ]
    )

def categorize_img(name: str):
    for category in CATEGORIES:
        if category in name:
            return category
    return "Unknown"


def normalize_color_space_img(img):
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] == 4:
            return ski.color.rgb2gray(ski.color.rgba2rgb(img))
        return ski.color.rgb2gray(img)
    raise ValueError(f"Unexpected image shape: {img.shape}")


def normalize_size(img, image_size: int):
    return ski.transform.resize(img, (image_size, image_size), anti_aliasing=True)


def preprocess_image(img):
    img_eq = exposure.equalize_adapthist(img, clip_limit=0.03)
    img_denoised = denoise_tv_chambolle(img_eq, weight=0.1)
    return img_denoised


def process_image_file(image_path: Path, image_size: int):
    raw_img = ski.io.imread(str(image_path))
    normalized = normalize_color_space_img(raw_img)
    resized = normalize_size(normalized, image_size)
    return preprocess_image(resized)


def load_binary_dataset(healthy_dir: Path, tumor_dir: Path, image_size: int):
    images = []
    labels = []

    for img_path in list_image_files(healthy_dir):
        images.append(process_image_file(img_path, image_size))
        labels.append(0)

    for img_path in list_image_files(tumor_dir):
        images.append(process_image_file(img_path, image_size))
        labels.append(1)

    return images, labels


def extract_texture_features(img):
    img_uint8 = (img * 255).astype(np.uint8)

    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(
        img_uint8,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )

    features = []
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    for prop in props:
        feat = graycoprops(glcm, prop)
        features.extend(feat.flatten())

    return np.array(features)


def extract_statistical_features(img):
    features = [
        np.mean(img),
        np.std(img),
        np.median(img),
        np.percentile(img, 25),
        np.percentile(img, 75),
        img.max() - img.min(),
    ]

    edges = filters.sobel(img)
    features.extend([np.mean(edges), np.std(edges)])

    return np.array(features)


def extract_all_features(img):
    hog_features = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=False,
    )

    texture_features = extract_texture_features(img)
    stat_features = extract_statistical_features(img)

    return np.concatenate([hog_features, texture_features, stat_features])


def main():
    healthy_dir = DATASET_HEALTHY_DIR
    tumor_dir = DATASET_TUMOR_DIR

    print("Loading and preprocessing training dataset...")
    images, labels = load_binary_dataset(healthy_dir, tumor_dir, IMAGE_SIZE)

    print("Extracting features from images...")
    x = np.array([extract_all_features(img) for img in images])
    y = np.array(labels)

    print(f"Feature vector size: {x.shape[1]}")
    print(f"Total samples: {x.shape[0]}")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x)
    y_train = y

    print(f"Training samples: {len(x_train)}")

    print("\nTraining SVM with fixed best hyperparameters...")
    print("Using: C=10, gamma='scale', kernel='rbf'")
    svm_clf = SVC(C=10, gamma="scale", kernel="rbf", random_state=RANDOM_STATE)
    svm_clf.fit(x_train, y_train)

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = MODEL_OUTPUT_DIR / "scaler.joblib"
    svm_path = MODEL_OUTPUT_DIR / "svm_model.joblib"

    joblib.dump(scaler, scaler_path)
    joblib.dump(svm_clf, svm_path)

    print("\nTraining complete. Exported artifacts:")
    print(f"- {scaler_path}")
    print(f"- {svm_path}")


if __name__ == "__main__":
    main()

