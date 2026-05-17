import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from src.utils.config import load_config
from src.models.encoder import get_encoder
from src.datasets.loader import ShenzhenDataset
from torch.utils.data import DataLoader, Subset
from src.models.proto_head import PrototypicalHead

print("Loading config...")
config = load_config()

print("Finding checkpoint...")
ckpt_dir = Path(config.logging.checkpoint_dir)
ckpts = list(ckpt_dir.glob("encoder_round_*.pt"))
ckpts.sort(key=lambda x: int(x.stem.split("_")[-1]))
latest_ckpt = ckpts[-1]
print(f"Latest checkpoint: {latest_ckpt}")

print("Loading encoder...")
encoder = get_encoder(config.model.backbone, config.model.embed_dim)
checkpoint = torch.load(latest_ckpt, map_location="cpu")
if "encoder_state_dict" in checkpoint:
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
else:
    encoder.load_state_dict(checkpoint)
encoder.eval()

print("Loading support set...")
support_ds = ShenzhenDataset(
    root_dir=config.data.shenzhen_path, 
    image_size=config.data.image_size
)

indices = []
c0, c1 = 0, 0
for i, label in enumerate(support_ds.labels):
    if label == 0 and c0 < 5:
        indices.append(i); c0 += 1
    if label == 1 and c1 < 5:
        indices.append(i); c1 += 1
    if c0 >= 5 and c1 >= 5: break

print(f"Indices: {indices}")
support_loader = DataLoader(Subset(support_ds, indices), batch_size=10)

with torch.no_grad():
    all_embeds = []
    all_labels = []
    for imgs, lbls in support_loader:
        embeds = encoder(imgs)
        all_embeds.append(embeds)
        all_labels.append(lbls)
    
    all_embeds = torch.cat(all_embeds)
    all_labels = torch.cat(all_labels)

print("\nEmbeddings Standard Deviation per feature:")
print(all_embeds.std(dim=0)[:10])
print("\nEmbeddings Mean per class (first 5 features):")
print(f"Class 0 (Normal): {all_embeds[all_labels==0].mean(dim=0)[:5]}")
print(f"Class 1 (TB):     {all_embeds[all_labels==1].mean(dim=0)[:5]}")

print("\nComputing prototypes...")
head = PrototypicalHead(config.model.embed_dim, num_classes=2)
head.compute_prototypes(all_embeds, all_labels)

print("\nPredicting on support set itself...")
labels, probs = head.predict(all_embeds)

for i in range(len(all_labels)):
    print(f"True: {all_labels[i].item()} | Pred: {labels[i].item()} | Prob(TB): {probs[i, 1].item():.4f}")

print("\nCosine similarities between Class 0 and Class 1 Prototypes:")
p0 = F.normalize(head.prototypes[0], p=2, dim=0)
p1 = F.normalize(head.prototypes[1], p=2, dim=0)
print(f"Sim: {torch.dot(p0, p1).item():.4f}")
