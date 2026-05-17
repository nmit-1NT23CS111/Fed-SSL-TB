import os
import sys
import json
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from pathlib import Path

# Add src to sys.path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config import load_config
from src.models.encoder import get_encoder
from src.models.proto_head import PrototypicalHead
from src.datasets.loader import get_eval_transform, ShenzhenDataset
from torch.utils.data import DataLoader, Subset

app = FastAPI(title="FedSSL TB Detection API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold models and config
config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = None
proto_head = None
transform = get_eval_transform(config.data.image_size)

@app.on_event("startup")
async def load_models():
    global encoder, proto_head
    print(f"Loading models to {device}...")
    
    # 1. Load Encoder dynamically (find latest round)
    encoder = get_encoder(config.model.backbone, config.model.embed_dim)
    ckpt_dir = Path(config.logging.checkpoint_dir)
    ckpts = list(ckpt_dir.glob("encoder_round_*.pt"))
    
    checkpoint = None
    if ckpts:
        # Sort by round number and pick the latest
        ckpts.sort(key=lambda x: int(x.stem.split("_")[-1]))
        latest_ckpt = ckpts[-1]
        
        checkpoint = torch.load(latest_ckpt, map_location=device)
        if "encoder_state_dict" in checkpoint:
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
        else:
            encoder.load_state_dict(checkpoint)
        print(f"Encoder dynamically loaded from {latest_ckpt}")
    else:
        print(f"WARNING: No checkpoints found in {ckpt_dir}. Using random weights.")
    
    encoder.to(device)
    encoder.eval()

    # 2. Initialize Prototypical Head
    proto_head = PrototypicalHead(config.model.embed_dim, num_classes=2)
    
    # 3. Load or Compute Prototypes
    if checkpoint and "prototypes" in checkpoint:
        proto_head.prototypes = checkpoint["prototypes"].to(device)
        proto_head._prototypes_computed = True
        print("Prototypes loaded from checkpoint.")
    else:
        print("Prototypes missing. Computing from Shenzhen support set...")
        try:
            # Load 5-shot support set to compute prototypes (FR3)
            support_ds = ShenzhenDataset(
                root_dir=config.data.shenzhen_path, 
                transform=transform,
                image_size=config.data.image_size
            )
            # Take first few samples per class
            indices = []
            c0, c1 = 0, 0
            for i, label in enumerate(support_ds.labels):
                if label == 0 and c0 < 5:
                    indices.append(i); c0 += 1
                if label == 1 and c1 < 5:
                    indices.append(i); c1 += 1
                if c0 >= 5 and c1 >= 5: break
            
            support_loader = DataLoader(Subset(support_ds, indices), batch_size=10)
            
            with torch.no_grad():
                all_embeds = []
                all_labels = []
                for imgs, lbls in support_loader:
                    embeds = encoder(imgs.to(device))
                    all_embeds.append(embeds)
                    all_labels.append(lbls.to(device))
                
                proto_head.compute_prototypes(
                    torch.cat(all_embeds), 
                    torch.cat(all_labels)
                )
            print(f"Prototypes computed successfully using {len(indices)} samples.")
        except Exception as e:
            print(f"ERROR computing prototypes: {e}")
            # Fallback to zeros to prevent crash, but classification will be random
            proto_head.prototypes = torch.zeros(2, config.model.embed_dim).to(device)
            proto_head._prototypes_computed = True
    
    proto_head.to(device)
    proto_head.eval()

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": str(device), "model": config.model.backbone}

@app.get("/metrics")
def get_training_metrics():
    log_path = Path(config.logging.log_dir) / "training_log.json"
    if not log_path.exists():
        return []
    with open(log_path, "r") as f:
        return json.load(f)

@app.post("/predict")
async def predict_tb(file: UploadFile = File(...)):
    if not encoder or not proto_head:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = encoder(img_tensor)
            labels, probs = proto_head.predict(embedding)
            
        prob_tb = float(probs[0, 1])
        status = "TB Positive" if prob_tb > 0.5 else "Normal"
        
        return {
            "filename": file.filename,
            "prediction": status,
            "confidence": round(prob_tb if prob_tb > 0.5 else (1 - prob_tb), 4),
            "tb_probability": round(prob_tb, 4),
            "normal_probability": round(1 - prob_tb, 4)
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
