import os
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import torch
import uvicorn 
from torchvision import transforms 
import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib

from config import config
from utils.database import db
from models.tabular_models import load_model as load_tabular_model
from models.cv_models import CVModelFactory

logger = logging.getLogger(__name__)

app = FastAPI(
    title=config.serving.api_title,
    description=config.serving.api_description,
    version=config.serving.api_version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_cache = {}
onnx_cache = {}

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    model_id: int
    data: Union[List[float], List[List[float]]]
    return_probabilities: bool = False

class PredictionResponse(BaseModel):
    predictions: Union[List[int], List[float], List[str]]
    probabilities: Optional[List[List[float]]] = None
    confidence: Optional[List[float]] = None
    model_info: Dict[str, Any]
    timestamp: str

class ModelInfo(BaseModel):
    id: int
    name: str
    model_type: str
    task_type: str
    metrics: Dict[str, Any]
    created_at: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: int
    database_status: str

# Dependency functions
def get_model_info(model_id: int) -> Dict[str, Any]:
    """Get model information from database"""
    models_df = db.get_models()
    model_row = models_df[models_df['id'] == model_id]

    if model_row.empty:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    return model_row.iloc[0].to_dict()


def load_cv_model(model_path: str, model_info: Dict[str, Any]) -> torch.nn.Module:
    """Load computer vision model"""
    try:
        task_type = model_info['task_type'].replace('cv_', '')

        num_classes = 10  # This should be stored in model metadata
        model = CVModelFactory.create_model(task_type, num_classes)

        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()

        return model
    except Exception as e:
        logger.error(f"Failed to load CV model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


def load_onnx_model(onnx_path: str) -> ort.InferenceSession:
    """Load ONNX model"""
    try:
        session = ort.InferenceSession(onnx_path)
        return session
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load ONNX model: {str(e)}")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for inference"""

    transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor.numpy()


# API Routes
@app.get("/", response_model=HealthResponse)
async def health_check():
    try:
        summary = db.get_experiment_summary()
        db_status = "healthy"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(model_cache),
        database_status=db_status
    )

@app.get("/models", response_model=List[ModelInfo])
async def list_models(task_type: Optional[str] = None, limit: int = 10):
    """List available models"""
    try:
        models_df = db.get_models(task_type=task_type)
        models_df = models_df.head(limit)

        models = []
        for _, row in models_df.iterrows():
            models.append(ModelInfo(
                id=row['id'],
                name=row['name'],
                model_type=row['model_type'],
                task_type=row['task_type'],
                metrics=eval(row['metrics']) if row['metrics'] else {},
                created_at=row['created_at']
            ))

        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: int):
    """Get specific model information"""
    model_info = get_model_info(model_id)

    return ModelInfo(
        id=model_info['id'],
        name=model_info['name'],
        model_type=model_info['model_type'],
        task_type=model_info['task_type'],
        metrics=eval(model_info['metrics']) if model_info['metrics'] else {},
        created_at=model_info['created_at']
    )


@app.post("/predict/tabular", response_model=PredictionResponse)
async def predict_tabular(request: PredictionRequest):
    """Make predictions on tabular data"""
    model_info = get_model_info(request.model_id)

    if not model_info['task_type'].startswith('tabular_'):
        raise HTTPException(status_code=400, detail="Model is not a tabular model")

    try:
        # Load model if not cached
        if request.model_id not in model_cache:
            model_path = model_info['file_path']
            if not Path(model_path).exists():
                raise HTTPException(status_code=404, detail="Model file not found")

            model = load_tabular_model(model_path)
            model_cache[request.model_id] = model
        else:
            model = model_cache[request.model_id]

        # Prepare data
        X = np.array(request.data)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = model.predict(X)

        probabilities = None
        confidence = None

        if request.return_probabilities and hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            if probs is not None:
                probabilities = probs.tolist()
                confidence = [max(prob) for prob in probs]

        # Log prediction
        db.log_prediction(
            model_id=request.model_id,
            input_data=request.data,
            prediction=predictions.tolist(),
            confidence=confidence[0] if confidence else None
        )

        return PredictionResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities,
            confidence=confidence,
            model_info={
                "model_id": request.model_id,
                "model_type": model_info['model_type'],
                "task_type": model_info['task_type']
            },
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    model_id: int = Form(...),
    use_onnx: bool = Form(False)
):
    """Make predictions on image data"""
    model_info = get_model_info(model_id)

    if not model_info['task_type'].startswith('cv_'):
        raise HTTPException(status_code=400, detail="Model is not a computer vision model")

    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Load and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        if use_onnx and model_info.get('onnx_path'):
            # Use ONNX model
            onnx_path = model_info['onnx_path']
            if not Path(onnx_path).exists():
                raise HTTPException(status_code=404, detail="ONNX model file not found")

            if model_id not in onnx_cache:
                session = load_onnx_model(onnx_path)
                onnx_cache[model_id] = session
            else:
                session = onnx_cache[model_id]

            # Preprocess for ONNX
            input_data = preprocess_image(image)

            # Run inference
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            result = session.run([output_name], {input_name: input_data})
            logits = result[0][0]

            # Convert to predictions
            predictions = int(np.argmax(logits))
            probabilities = torch.softmax(torch.tensor(logits), dim=0).numpy().tolist()
            confidence = float(max(probabilities))

        else:
            # Use PyTorch model
            if model_id not in model_cache:
                model_path = model_info['file_path']
                if not Path(model_path).exists():
                    raise HTTPException(status_code=404, detail="Model file not found")

                model = load_cv_model(model_path, model_info)
                model_cache[model_id] = model
            else:
                model = model_cache[model_id]

            # Preprocess image
            input_tensor = preprocess_image(image)
            input_tensor = torch.tensor(input_tensor)

            # Make prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0].numpy().tolist()
                predictions = int(torch.argmax(outputs, dim=1).item())
                confidence = float(max(probabilities))

        # Log prediction
        db.log_prediction(
            model_id=model_id,
            input_data={"filename": file.filename, "size": len(image_data)},
            prediction=predictions,
            confidence=confidence
        )

        return JSONResponse({
            "predictions": predictions,
            "probabilities": probabilities,
            "confidence": confidence,
            "model_info": {
                "model_id": model_id,
                "model_type": model_info['model_type'],
                "task_type": model_info['task_type'],
                "used_onnx": use_onnx
            },
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Image prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch/tabular")
async def predict_batch_tabular(
    model_id: int = Form(...),
    file: UploadFile = File(...)
):
    """Make batch predictions on tabular data from CSV file"""
    model_info = get_model_info(model_id)

    if not model_info['task_type'].startswith('tabular_'):
        raise HTTPException(status_code=400, detail="Model is not a tabular model")

    try:
        # Load model
        if model_id not in model_cache:
            model_path = model_info['file_path']
            if not Path(model_path).exists():
                raise HTTPException(status_code=404, detail="Model file not found")

            model = load_tabular_model(model_path)
            model_cache[model_id] = model
        else:
            model = model_cache[model_id]

        # Read CSV file
        file_data = await file.read()
        df = pd.read_csv(io.StringIO(file_data.decode('utf-8')))

        # Make predictions
        X = df.values
        predictions = model.predict(X)

        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            if probs is not None:
                probabilities = probs.tolist()

        # Create results DataFrame
        results_df = df.copy()
        results_df['prediction'] = predictions

        if probabilities:
            for i, class_name in enumerate(model.target_names or [f"class_{i}" for i in range(len(probabilities[0]))]):
                results_df[f'prob_{class_name}'] = [prob[i] for prob in probabilities]

        return JSONResponse({
            "predictions": predictions.tolist(),
            "probabilities": probabilities,
            "results": results_df.to_dict('records'),
            "model_info": {
                "model_id": model_id,
                "model_type": model_info['model_type'],
                "task_type": model_info['task_type']
            },
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/experiments")
async def list_experiments(task_type: Optional[str] = None, limit: int = 10):
    """List experiments"""
    try:
        experiments_df = db.get_experiments(task_type=task_type)
        experiments_df = experiments_df.head(limit)

        experiments = []
        for _, row in experiments_df.iterrows():
            experiments.append({
                "id": row['id'],
                "name": row['name'],
                "task_type": row['task_type'],
                "status": row['status'],
                "metrics": eval(row['metrics']) if row['metrics'] else {},
                "created_at": row['created_at']
            })

        return experiments
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        summary = db.get_experiment_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache")
async def clear_cache():
    """Clear model cache"""
    global model_cache, onnx_cache
    model_cache.clear()
    onnx_cache.clear()

    return {"message": "Cache cleared successfully"}


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting AutoML API server...")

    # Ensure model directories exist
    Path(config.serving.model_dir).mkdir(parents=True, exist_ok=True)
    Path(config.serving.onnx_dir).mkdir(parents=True, exist_ok=True)

    logger.info("AutoML API server started successfully")


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=config.serving.host,
        port=config.serving.port,
        reload=config.serving.reload
    )
