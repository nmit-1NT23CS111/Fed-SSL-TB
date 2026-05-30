"""
src/web/api.py
--------------
FastAPI backend for Federated SSL TB Detection.

Startup sequence:
  If linear_head.pt exists (run run_finetune.py first):
    - Mode A: ImageNet ViT-Tiny encoder + trained linear head  [ACCURATE]
  Else:
    - Mode B: FL MAE encoder + 5-shot prototypical             [FALLBACK]
"""

import sys
import json
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config import load_config
from src.models.encoder import get_encoder
from src.models.proto_head import PrototypicalHead
from src.datasets.loader import get_eval_transform, ShenzhenDataset
from torch.utils.data import DataLoader, Subset

app = FastAPI(title="FedSSL TB Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

config    = load_config()
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_eval_transform(config.data.image_size)

# Runtime globals
encoder    = None
classifier = None
use_linear = False


def _load_pretrained_vit():
    import timm
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("[Startup] ImageNet ViT-Tiny loaded (embed_dim=192)")
    return model


def _load_fl_encoder():
    enc      = get_encoder(config.model.backbone, config.model.embed_dim)
    ckpt_dir = Path(config.logging.checkpoint_dir)
    ckpts    = sorted(ckpt_dir.glob("encoder_round_*.pt"),
                      key=lambda x: int(x.stem.split("_")[-1]))
    if ckpts:
        latest = ckpts[-1]
        ckpt   = torch.load(latest, map_location=device, weights_only=False)
        state  = ckpt["encoder_state_dict"] if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt else ckpt
        enc.load_state_dict(state)
        print(f"[Startup] FL encoder loaded: {latest.name}")
    else:
        print("[Startup] WARNING: No FL checkpoint found, using random encoder")
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


@app.on_event("startup")
async def load_models():
    global encoder, classifier, use_linear

    print(f"[Startup] Device: {device}")
    ckpt_dir         = Path(config.logging.checkpoint_dir)
    linear_head_path = ckpt_dir / "linear_head.pt"

    if linear_head_path.exists():
        # ── Mode A: Pretrained ViT-Tiny + linear head ─────────────────────────
        print("[Startup] Mode A: ImageNet ViT-Tiny + Linear Probe")
        encoder = _load_pretrained_vit().to(device)

        head = nn.Linear(192, 2)
        head.load_state_dict(torch.load(linear_head_path, map_location=device, weights_only=False))
        head.to(device).eval()
        classifier = head
        use_linear = True
        print("[Startup] Ready (Linear Probe mode)")

    else:
        # ── Mode B: FL encoder + 5-shot prototypical ──────────────────────────
        print("[Startup] Mode B: FL encoder + 5-shot Prototypical (run run_finetune.py for better accuracy)")
        encoder = _load_fl_encoder().to(device)

        proto = PrototypicalHead(embed_dim=config.model.embed_dim, num_classes=2)
        try:
            ds = ShenzhenDataset(
                root_dir=config.data.shenzhen_path,
                transform=transform,
                image_size=config.data.image_size,
            )
            k = config.finetuning.few_shot_k
            idx, c0, c1 = [], 0, 0
            for i, lbl in enumerate(ds.labels):
                if lbl == 0 and c0 < k:   idx.append(i); c0 += 1
                elif lbl == 1 and c1 < k:  idx.append(i); c1 += 1
                if c0 >= k and c1 >= k:    break

            loader = DataLoader(Subset(ds, idx), batch_size=len(idx), shuffle=False)
            with torch.no_grad():
                imgs, lbls = next(iter(loader))
                embs = encoder(imgs.to(device))
                proto.compute_prototypes(embs, lbls.to(device))
                _, probs = proto.predict(embs)
                acc = (probs.argmax(-1).cpu() == lbls).float().mean().item()
            print(f"[Startup] Prototypes ready — support self-accuracy: {acc:.1%}")
        except Exception as e:
            print(f"[Startup] Prototype ERROR: {e}")
            import traceback; traceback.print_exc()

        proto.to(device).eval()
        classifier = proto
        use_linear = False
        print("[Startup] Ready (Prototypical mode)")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "model": config.model.backbone,
        "classifier": "linear_probe" if use_linear else "prototypical_5shot",
    }


@app.get("/metrics")
def get_training_metrics():
    log_path = Path(config.logging.log_dir) / "training_log.json"
    if not log_path.exists():
        return []
    with open(log_path, "r") as f:
        return json.load(f)


@app.post("/predict")
async def predict_tb(file: UploadFile = File(...)):
    if encoder is None or classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        contents   = await file.read()
        image      = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = encoder(img_tensor)
            if use_linear:
                probs = torch.softmax(classifier(emb), dim=-1)
            else:
                _, probs = classifier.predict(emb)

        prob_tb     = float(probs[0, 1])
        prob_normal = float(probs[0, 0])
        # Medical AI: prefer sensitivity — lower threshold catches more TB cases
        status      = "TB Positive" if prob_tb > 0.35 else "Normal"
        confidence  = round(prob_tb if prob_tb > 0.35 else prob_normal, 4)

        return {
            "filename":           file.filename,
            "prediction":         status,
            "confidence":         round(max(prob_tb, prob_normal), 4),
            "tb_probability":     round(prob_tb, 4),
            "normal_probability": round(prob_normal, 4),
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
