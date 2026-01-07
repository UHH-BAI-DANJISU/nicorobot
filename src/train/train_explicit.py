import torch
import csv
from torch.utils.data import DataLoader
from datasets.nico_dataset import NicoDataset
from models.explicit_bc import ExplicitBC

device = "cuda" if torch.cuda.is_available() else "cpu"

train_ds = NicoDataset("data/nico/real_evo_ik_samples")
test_ds  = NicoDataset("data/nico/real_evo_ik_samples_test")

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=128)

model = ExplicitBC(action_dim=train_ds[0][1].shape[0]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

log_path = "explicit_loss.csv"
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss"])

    for epoch in range(30):
        model.train()
        total = 0
        for obs, action in train_loader:
            obs, action = obs.to(device), action.to(device)
            pred = model(obs)
            loss = loss_fn(pred, action)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        avg_loss = total / len(train_loader)
        writer.writerow([epoch, avg_loss])
        f.flush()

        print(f"[Explicit][Epoch {epoch}] loss = {avg_loss:.4f}")

print(f"Saved explicit results to {log_path}")
