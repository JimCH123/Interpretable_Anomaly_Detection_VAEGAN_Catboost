"""Real-time anomaly detection API (dissertation Section IV-A, online validation).

``POST /predict`` takes one raw 590-value record as a comma-separated line,
applies the offline preprocessing, scores it with CatBoost and, when the record is
flagged as anomalous, renders a TreeSHAP waterfall plot that can be fetched from
``GET /plot/{file}``.

Run with ``python scripts/06_serve_api.py`` or ``uvicorn anomaly_detection.api:app``.
"""
import logging
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config
from .preprocessing import selected_feature_names, transform

logger = logging.getLogger(__name__)

config.API_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ----- 1) Model & Explainer -----
model = CatBoostClassifier()
model.load_model(config.MODEL)
threshold = float(np.load(config.THRESHOLD))
background = np.load(config.BACKGROUND)
explainer = shap.TreeExplainer(model, background)
FEATURE_NAMES = selected_feature_names()

# ----- 2) FastAPI -----
app = FastAPI(title="UCI-SECOM Anomaly Detection API")


class LineIn(BaseModel):
    line: str


class ResultOut(BaseModel):
    score: float
    is_anomaly: bool
    plot_file: Optional[str] = None


def generate_waterfall_plot(shap_val, score, plot_fp):
    plt.figure()
    shap.plots.waterfall(shap_val, show=False)
    plt.title(f"score={score:.3f}")
    plt.savefig(plot_fp, bbox_inches="tight")
    plt.close()


def run_inference(x: np.ndarray):
    """Return ``(score, is_anomaly, plot file name or None)`` for one transformed record."""
    try:
        score = float(model.predict_proba(x)[0, 1])
        is_abn = score >= threshold
        plot_nm = None
        if is_abn:
            x_df = pd.DataFrame(x, columns=FEATURE_NAMES)
            shap_val = explainer(x_df)
            ts = datetime.now().strftime("%Y%m%dT%H%M%S%f")
            plot_nm = f"{ts}.png"
            generate_waterfall_plot(shap_val[0], score, config.API_PLOT_DIR / plot_nm)
        return score, is_abn, plot_nm
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ResultOut)
def predict(item: LineIn):
    try:
        raw_list = item.line.split(",")
        x_std = transform(raw_list)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    score, is_abn, plot_nm = run_inference(x_std)
    return {"score": score, "is_anomaly": is_abn, "plot_file": plot_nm}


@app.get("/plot/{file}")
def get_plot(file: str):
    fp = config.API_PLOT_DIR / file
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(fp, media_type="image/png")
