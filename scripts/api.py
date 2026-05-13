from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import joblib
import json
import re
from pydantic import BaseModel

from io import BytesIO
from pathlib import Path
import numpy as np
import skimage as ski
from skimage import exposure, filters
from skimage.feature import graycomatrix, graycoprops, hog
from skimage.restoration import denoise_tv_chambolle

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to load existing unsupervised artifacts if present (non-fatal)
try:
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("kmeans_model.pkl")
    with open("cluster_keywords.json") as f:
        cluster_keywords = json.load(f)
except Exception:
    vectorizer = None
    model = None
    cluster_keywords = {}

# Paths and config for supervised SVM model (trained script exports to ./models)
SCALER_PATH = Path("./models/scaler.joblib")
SVM_PATH = Path("./models/svm_model.joblib")
IMAGE_SIZE = 256

# Lazy-load scaler and svm
_scaler = None
_svm = None


def load_supervised_artifacts():
    global _scaler, _svm
    if _scaler is None and SCALER_PATH.exists():
        _scaler = joblib.load(str(SCALER_PATH))
    if _svm is None and SVM_PATH.exists():
        _svm = joblib.load(str(SVM_PATH))


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


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request, "result": None})


@app.post("/predict-image", response_class=HTMLResponse)
async def predict_image(request: Request, file: UploadFile = File(...)):
    load_supervised_artifacts()
    if _scaler is None or _svm is None:
        return templates.TemplateResponse(
            "upload.html",
            {"request": request, "result": "Model artifacts not found. Train and export models to ./models."},
        )

    contents = await file.read()
    try:
        img = ski.io.imread(BytesIO(contents))
    except Exception:
        return templates.TemplateResponse(
            "upload.html",
            {"request": request, "result": "Failed to read image. Please upload a valid image file."},
        )

    try:
        img_gray = normalize_color_space_img(img)
        img_resized = normalize_size(img_gray, IMAGE_SIZE)
        img_pre = preprocess_image(img_resized)
        features = extract_all_features(img_pre)
        x = _scaler.transform([features])
        pred = int(_svm.predict(x)[0])
        label = "Tumor" if pred == 1 else "Healthy"
        return templates.TemplateResponse(
            "upload.html",
            {"request": request, "result": label},
        )
    except Exception as e:
        return templates.TemplateResponse(
            "upload.html",
            {"request": request, "result": f"Error during processing: {e}"},
        )


@app.get("/")
def home():
    return {"status": "API running"}

