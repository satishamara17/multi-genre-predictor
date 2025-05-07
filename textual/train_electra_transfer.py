import argparse, random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    ElectraTokenizerFast,
    ElectraModel,
    get_linear_schedule_with_warmup,
)

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Electra-small on multi-label genres")
    p.add_argument("--tsv",           required=True, help="Input TSV (plot\tgenres)")
    p.add_argument("--epochs",   type=int,   default=5)
    p.add_argument("--batch",    type=int,   default=8)
    p.add_argument("--accum",    type=int,   default=1,    help="Gradient accumulation steps")
    p.add_argument("--lr",       type=float, default=2e-5)
    p.add_argument("--max_len",  type=int,   default=384)
    p.add_argument("--freeze_layers", type=int, default=4,
                   help="Number of bottom Electra layers to freeze")
    p.add_argument("--warmup_ratio",  type=float, default=0.1)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--out",      type=str,   default="electra_transfer.pth")
    return p.parse_args()

args = parse_args()

# reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                     else "cuda" if torch.cuda.is_available() else "cpu")
print("→ Device:", DEVICE)

# 1) LOAD & PREP DATAFRAME
df = pd.read_csv(
    args.tsv,
    sep="\t",
    engine="python",
    quoting=3,
    on_bad_lines="skip",
)
df = df.dropna(subset=["plot"]).reset_index(drop=True)

def parse_genres(s):
    if isinstance(s, str):
        return [g.strip() for g in s.split("|") if g.strip()]
    return []

df["genres_list"] = df["genres"].apply(parse_genres)
df = df[df["genres_list"].map(len) > 0].reset_index(drop=True)

# 2) SPLIT 80/10/10
idx = list(range(len(df)))
random.shuffle(idx)
n = len(idx)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)
train_df = df.iloc[idx[:n_train]]
val_df   = df.iloc[idx[n_train:n_train+n_val]]
test_df  = df.iloc[idx[n_train+n_val:]]

print(f"→ Split: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

# 3) LABELS
LABELS = ["Action","Comedy","Adventure","Thriller",
          "Crime","Drama","Horror","Romance","Sci-Fi"]
lab2id = {g:i for i,g in enumerate(LABELS)}
N_LABELS = len(LABELS)

# 4) DATASET
class MovieDataset(Dataset):
    def __init__(self, df):
        self.texts  = df["plot"].tolist()
        self.labels = df["genres_list"].tolist()
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        text = self.texts[i]
        y = torch.zeros(N_LABELS, dtype=torch.float)
        for g in self.labels[i]:
            if g in lab2id:
                y[lab2id[g]] = 1
        return text, y

# 5) TOKENIZER + COLLATE_FN
tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator")

def collate_batch(batch):
    texts, labels = zip(*batch)
    enc = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=args.max_len,
        return_tensors="pt"
    )
    enc = {k:v.to(DEVICE) for k,v in enc.items()}
    enc["labels"] = torch.stack(labels).to(DEVICE)
    return enc

train_loader = DataLoader(
    MovieDataset(train_df),
    batch_size=args.batch,
    shuffle=True,
    collate_fn=collate_batch
)
val_loader = DataLoader(
    MovieDataset(val_df),
    batch_size=args.batch,
    shuffle=False,
    collate_fn=collate_batch
)
test_loader = DataLoader(
    MovieDataset(test_df),
    batch_size=args.batch,
    shuffle=False,
    collate_fn=collate_batch
)

# 6) MODEL
class ElectraMultiLabel(nn.Module):
    def __init__(self, freeze_layers:int):
        super().__init__()
        self.backbone = ElectraModel.from_pretrained(
            "google/electra-small-discriminator"
        )
        # freeze bottom-k layers
        for i, layer in enumerate(self.backbone.encoder.layer):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False
        self.classifier = nn.Linear(self.backbone.config.hidden_size, N_LABELS)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:,0]
        return self.classifier(cls)

model = ElectraMultiLabel(args.freeze_layers).to(DEVICE)

# 7) OPTIMIZER & SCHEDULER
optimizer = AdamW(model.parameters(), lr=args.lr)
total_steps = len(train_loader) * args.epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(args.warmup_ratio * total_steps),
    num_training_steps=total_steps
)
criterion = nn.BCEWithLogitsLoss()

# 8) METRIC HELPERS
def prf(probs, labels, t):
    preds = (probs >= t).int()
    y     = (labels >= 0.5).int()
    eps   = 1e-9
    tp = (preds & y).sum(dim=0).float()
    fp = (preds & ~y).sum(dim=0).float()
    fn = ((~preds) & y).sum(dim=0).float()
    prec = (tp/(tp+fp+eps)).mean().item()
    rec  = (tp/(tp+fn+eps)).mean().item()
    f1   = 2*prec*rec/(prec+rec+eps)
    return prec, rec, f1

def sweep(probs, labels):
    best = (0,0,0,-1)
    for t in np.arange(0.0, 0.55, 0.05):
        p,r,f = prf(probs, labels, t)
        if f > best[3]:
            best = (t,p,r,f)
    return best

# 9) TRAIN / EVAL LOOPS
def train_one_epoch():
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(train_loader, start=1):
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss   = criterion(logits, batch["labels"]) / args.accum
        loss.backward()
        running_loss += loss.item()
        if step % args.accum == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    return running_loss / len(train_loader)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    all_p, all_y = [], []
    for batch in loader:
        logits = model(batch["input_ids"], batch["attention_mask"])
        total_loss += criterion(logits, batch["labels"]).item()
        all_p.append(torch.sigmoid(logits).cpu())
        all_y.append(batch["labels"].cpu())
    return (
        total_loss / len(loader),
        torch.cat(all_p, dim=0),
        torch.cat(all_y, dim=0)
    )

# 10) MAIN TRAINING
train_loss =[]
validation_loss =[]
best_f1, best_t = -1, 0
for epoch in range(1, args.epochs+1):
    tr_loss = train_one_epoch()
    train_loss.append(tr_loss)
    val_loss, val_p, val_y = evaluate(val_loader)
    validation_loss.append(val_loss)
    t, p, r, f = sweep(val_p, val_y)
    print(f"[{epoch}/{args.epochs}] train={tr_loss:.3f}  val={val_loss:.3f}  "
          f"t={t:.2f}  P={p:.3f}  R={r:.3f}  F1={f:.3f}")
    if f > best_f1:
        best_f1, best_t = f, t
        torch.save(model.state_dict(), args.out)
        print("  ↳ Saved new best checkpoint")

print(f"\n Best val F1={best_f1:.3f} @ thresh={best_t:.2f}")

# 11) FINAL TEST
model.load_state_dict(torch.load(args.out))
test_loss, test_p, test_y = evaluate(test_loader)
p, r, f = prf(test_p, test_y, best_t)
print(f"[TEST] P={p:.3f}  R={r:.3f}  F1={f:.3f}")


epochs = range(1, len(train_loss) + 1)
import matplotlib.pyplot as plt
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, validation_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()