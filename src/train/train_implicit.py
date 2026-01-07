import torch
import csv
from torch.utils.data import DataLoader
from datasets.nico_dataset import NicoDataset
from models.energy_model import EnergyModel
from losses.info_nce import info_nce_loss
from utils.sampling import sample_negative_actions

device = "cuda" if torch.cuda.is_available() else "cpu"

train_ds = NicoDataset("data/nico/real_evo_ik_samples")
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

action_dim = train_ds[0][1].shape[0]
model = EnergyModel(action_dim).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

log_path = "implicit_loss.csv"
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss"])

    for epoch in range(30):
        total = 0
        for obs, action in train_loader:
            obs, action = obs.to(device), action.to(device)

            neg = sample_negative_actions(action)
            B, K, D = neg.shape

            obs_rep = obs.unsqueeze(1).repeat(1, K, 1, 1, 1).view(B*K, 6, 64, 64)
            neg = neg.view(B*K, D)

            e_pos = model(obs, action)
            e_neg = model(obs_rep, neg).view(B, K)

            loss = info_nce_loss(e_pos, e_neg)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        avg_loss = total / len(train_loader)
        writer.writerow([epoch, avg_loss])
        f.flush()

        print(f"[Implicit][Epoch {epoch}] loss = {avg_loss:.4f}")

print(f"Saved implicit results to {log_path}")
