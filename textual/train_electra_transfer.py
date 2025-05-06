# #!/usr/bin/env python
# """
# train_electra_small.py
# Fine-tune google/electra-small-discriminator for multi-label genre classification.
# """

# import argparse, random, os, csv
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import (
#     ElectraTokenizerFast,
#     ElectraModel,
#     DataCollatorWithPadding,
#     AdamW,
#     get_linear_schedule_with_warmup
# )

# # -----------------------------------------------------------------------------
# # 1) CLI arguments
# # -----------------------------------------------------------------------------
# def parse_args():
#     p = argparse.ArgumentParser(description="Fine-tune Electra-small for multi-label classification")
#     p.add_argument("--tsv",      required=True, help="Path to filtered TSV (plot|genres)")
#     p.add_argument("--epochs",   type=int,   default=5)
#     p.add_argument("--batch",    type=int,   default=8)
#     p.add_argument("--accum",    type=int,   default=1,   help="Gradient accumulation steps")
#     p.add_argument("--lr",       type=float, default=2e-5)
#     p.add_argument("--max_len",  type=int,   default=384)
#     p.add_argument("--warmup",   type=float, default=0.1, help="Warmup ratio")
#     p.add_argument("--seed",     type=int,   default=42)
#     p.add_argument("--out",      default="electra_best.pth")
#     return p.parse_args()

# args = parse_args()

# # -----------------------------------------------------------------------------
# # 2) Repro & device
# # -----------------------------------------------------------------------------
# torch.manual_seed(args.seed)
# random.seed(args.seed)
# np.random.seed(args.seed)

# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
#                       "cuda" if torch.cuda.is_available() else "cpu")
# print("→ Device:", DEVICE)

# # -----------------------------------------------------------------------------
# # 3) Labels & Data load
# # -----------------------------------------------------------------------------
# LABELS = ["Action","Comedy","Adventure","Thriller",
#           "Crime","Drama","Horror","Romance","Sci-Fi"]
# lab2id = {g:i for i,g in enumerate(LABELS)}
# N_LABELS = len(LABELS)

# df = pd.read_csv(args.tsv, sep="\t", quoting=csv.QUOTE_NONE,
#                  engine="python", on_bad_lines="skip")
# print(f"→ loaded {len(df):,} rows")

# # drop any missing
# df = df.dropna(subset=["plot","genres"]).reset_index(drop=True)

# # parse pipe-delimited genre lists
# def parse_list(s):
#     return [g.strip() for g in s.split("|") if g.strip()]

# df["tags"] = df["genres"].map(parse_list)

# # shuffle & split 80/10/10
# idx = list(range(len(df)))
# random.shuffle(idx)
# n = len(idx)
# n1 = int(0.8*n)
# n2 = int(0.9*n)
# train_df = df.iloc[idx[:n1]].reset_index(drop=True)
# val_df   = df.iloc[idx[n1:n2]].reset_index(drop=True)
# test_df  = df.iloc[idx[n2: ]].reset_index(drop=True)
# print(f"→ splits: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

# # -----------------------------------------------------------------------------
# # 4) Tokenizer, collator, dataset
# # -----------------------------------------------------------------------------
# tok = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator")
# collator = DataCollatorWithPadding(tok, return_tensors="pt")

# class MovieDS(Dataset):
#     def __init__(self, df):
#         self.texts  = df["plot"].tolist()
#         self.tags   = df["tags"].tolist()
#     def __len__(self): return len(self.texts)
#     def __getitem__(self, i):
#         text = str(self.texts[i])
#         enc  = tok(text,
#                    truncation=True,
#                    max_length=args.max_len,
#                    return_tensors="pt")
#         y = torch.zeros(N_LABELS, dtype=torch.float)
#         for g in self.tags[i]:
#             if g in lab2id: y[lab2id[g]] = 1.0
#         # squeeze batch dim:
#         item = {k: v.squeeze(0) for k,v in enc.items()}
#         item["labels"] = y
#         return item

# dl_train = DataLoader(MovieDS(train_df), batch_size=args.batch,
#                       shuffle=True, collate_fn=collator)
# dl_val   = DataLoader(MovieDS(val_df),   batch_size=args.batch,
#                       shuffle=False, collate_fn=collator)
# dl_test  = DataLoader(MovieDS(test_df),  batch_size=args.batch,
#                       shuffle=False, collate_fn=collator)

# # -----------------------------------------------------------------------------
# # 5) Model definition
# # -----------------------------------------------------------------------------
# class ElectraMulti(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.backbone = ElectraModel.from_pretrained("google/electra-small-discriminator")
#         self.head = nn.Linear(self.backbone.config.hidden_size, N_LABELS)
#     def forward(self, input_ids, attention_mask):
#         out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
#         cls = out.last_hidden_state[:,0]
#         return self.head(cls)

# model = ElectraMulti().to(DEVICE)

# # -----------------------------------------------------------------------------
# # 6) Optimizer & scheduler
# # -----------------------------------------------------------------------------
# optimizer = AdamW(model.parameters(), lr=args.lr)
# total_steps = len(dl_train) * args.epochs // args.accum
# warmup_steps = int(args.warmup * total_steps)
# scheduler = get_linear_schedule_with_warmup(optimizer,
#                                             num_warmup_steps=warmup_steps,
#                                             num_training_steps=total_steps)

# criterion = nn.BCEWithLogitsLoss()

# # -----------------------------------------------------------------------------
# # 7) Metrics helpers
# # -----------------------------------------------------------------------------
# def prf(probs, y, t):
#     p = (probs>=t).int(); y = y.int()
#     eps = 1e-9
#     tp = (p & y).sum(dim=0).float()
#     fp = (p & ~y).sum(dim=0).float()
#     fn = ((~p)& y).sum(dim=0).float()
#     prec = (tp/(tp+fp+eps)).mean().item()
#     rec  = (tp/(tp+fn+eps)).mean().item()
#     f1   = 2*prec*rec/(prec+rec+eps)
#     return prec, rec, f1

# def sweep(probs, y):
#     best = (0,0,0,-1.0)
#     for t in np.arange(0,0.55,0.05):
#         p,r,f = prf(probs, y, t)
#         if f>best[3]: best=(t,p,r,f)
#     return best

# # -----------------------------------------------------------------------------
# # 8) Train / Eval loops
# # -----------------------------------------------------------------------------
# def run_epoch(dl, train=False):
#     model.train() if train else model.eval()
#     torch.set_grad_enabled(train)
#     total_loss = 0
#     all_p, all_y = [], []
#     for step, batch in enumerate(dl, 1):
#         input_ids = batch["input_ids"].to(DEVICE)
#         mask      = batch["attention_mask"].to(DEVICE)
#         labels    = batch["labels"].to(DEVICE)

#         logits = model(input_ids, mask)
#         loss   = criterion(logits, labels)
#         if train:
#             loss = loss / args.accum
#             loss.backward()
#             if step % args.accum == 0:
#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()
#         total_loss += loss.item() * args.accum
#         all_p.append(torch.sigmoid(logits).cpu())
#         all_y.append(labels.cpu())

#     avg_loss = total_loss/len(dl)
#     return avg_loss, torch.cat(all_p), torch.cat(all_y)

# best_f1, best_t = -1.0, 0.0
# for ep in range(1, args.epochs+1):
#     tr_loss, _, _ = run_epoch(dl_train, train=True)
#     v_loss, v_p, v_y = run_epoch(dl_val, train=False)
#     t,p,r,f = sweep(v_p, v_y)
#     print(f"[{ep}/{args.epochs}] train={tr_loss:.3f} val={v_loss:.3f}  → t={t:.2f} P={p:.3f} R={r:.3f} F1={f:.3f}")
#     if f>best_f1:
#         best_f1, best_t = f,t
#         torch.save(model.state_dict(), args.out)
#         print("  ↳ new best saved")

# print(f"\n▶︎ BEST VAL F1={best_f1:.3f} @ t={best_t:.2f}")

# # final test
# model.load_state_dict(torch.load(args.out))
# _, te_p, te_y = run_epoch(dl_test, train=False)
# tp, tr, tf = prf(te_p, te_y, best_t)
# print(f"[TEST] Precision={tp:.3f} Recall={tr:.3f} F1={tf:.3f}")

#!/usr/bin/env python
import argparse, random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    ElectraTokenizerFast,
    ElectraModel,
    AdamW,
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
    quoting=3,            # QUOTE_NONE
    on_bad_lines="skip",  # skip malformed lines
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
best_f1, best_t = -1, 0
for epoch in range(1, args.epochs+1):
    tr_loss = train_one_epoch()
    val_loss, val_p, val_y = evaluate(val_loader)
    t, p, r, f = sweep(val_p, val_y)
    print(f"[{epoch}/{args.epochs}] train={tr_loss:.3f}  val={val_loss:.3f}  "
          f"t={t:.2f}  P={p:.3f}  R={r:.3f}  F1={f:.3f}")
    if f > best_f1:
        best_f1, best_t = f, t
        torch.save(model.state_dict(), args.out)
        print("  ↳ Saved new best checkpoint")

print(f"\n▶︎ Best val F1={best_f1:.3f} @ thresh={best_t:.2f}")

# 11) FINAL TEST
model.load_state_dict(torch.load(args.out))
test_loss, test_p, test_y = evaluate(test_loader)
p, r, f = prf(test_p, test_y, best_t)
print(f"[TEST] P={p:.3f}  R={r:.3f}  F1={f:.3f}")

# #!/usr/bin/env python
# import argparse, random, math, csv
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from collections import Counter
# from transformers import (
#     ElectraTokenizerFast,
#     ElectraModel,
#     DataCollatorWithPadding,
#     AdamW,
# )
# from torch.optim.lr_scheduler import OneCycleLR

# # ─── CLI ───────────────────────────────────────────────────────────────────────
# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--tsv",          required=True,
#                    help="Tab-separated file with columns ['plot','genres']")
#     p.add_argument("--epochs",   type=int,   default=5)
#     p.add_argument("--batch",    type=int,   default=8)
#     p.add_argument("--lr_backbone",  type=float, default=5e-6,
#                    help="LR for Electra backbone")
#     p.add_argument("--lr_head",      type=float, default=2e-5,
#                    help="LR for classifier & dropout/LN head")
#     p.add_argument("--warmup_ratio", type=float, default=0.1)
#     p.add_argument("--max_len",   type=int,   default=384)
#     p.add_argument("--freeze_layers", type=int, default=4,
#                    help="How many bottom Electra layers to freeze")
#     p.add_argument("--seed",     type=int,   default=42)
#     p.add_argument("--out",      default="electra_refined.pth")
#     return p.parse_args()

# args = parse_args()

# # ─── REPRO ─────────────────────────────────────────────────────────────────────
# torch.manual_seed(args.seed)
# random.seed(args.seed)
# np.random.seed(args.seed)

# DEVICE = torch.device(
#     "mps" if torch.backends.mps.is_available()
#     else "cuda" if torch.cuda.is_available()
#     else "cpu"
# )
# print(f"→ Device: {DEVICE}")

# # ─── LABELS ────────────────────────────────────────────────────────────────────
# LABELS = ["Action","Comedy","Adventure","Thriller",
#           "Crime","Drama","Horror","Romance","Sci-Fi"]
# lab2id = {g:i for i,g in enumerate(LABELS)}
# N_LABELS = len(LABELS)

# # ─── DATA ──────────────────────────────────────────────────────────────────────
# def parse_tags(s: str):
#     return [g.strip() for g in s.split("|") if g.strip()]

# df = pd.read_csv(
#     args.tsv,
#     sep="\t",
#     quoting=csv.QUOTE_NONE,
#     engine="python",
#     on_bad_lines="skip"
# ).dropna(subset=["plot"])
# df["tags"] = df["genres"].apply(parse_tags)
# print(f"→ loaded {len(df):,} rows")

# # 80/10/10 shuffle split
# idx = np.random.permutation(len(df))
# n = len(idx)
# train_df = df.iloc[idx[:int(0.8*n)]].reset_index(drop=True)
# val_df   = df.iloc[idx[int(0.8*n):int(0.9*n)]].reset_index(drop=True)
# test_df  = df.iloc[idx[int(0.9*n):]].reset_index(drop=True)
# print(f"→ split: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

# # ─── TOKENIZER & DATALOADER ───────────────────────────────────────────────────
# tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator")
# collator  = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# class MovieDS(Dataset):
#     def __init__(self, df):
#         self.texts = df["plot"].tolist()
#         self.tags  = df["tags"].tolist()
#     def __len__(self): return len(self.texts)
#     def __getitem__(self, i):
#         return {
#             "text": self.texts[i],
#             "labels": torch.tensor(
#                 [1 if g in self.tags[i] else 0 for g in LABELS],
#                 dtype=torch.float
#             )
#         }

# def make_loader(df, shuffle=False):
#     ds = MovieDS(df)
#     def collate(batch):
#         texts  = [b["text"] for b in batch]
#         labs   = torch.stack([b["labels"] for b in batch])
#         enc    = tokenizer(
#                      texts,
#                      truncation=True,
#                      max_length=args.max_len,
#                      padding=True,
#                      return_tensors="pt"
#                  )
#         return {
#             "input_ids":      enc.input_ids,
#             "attention_mask": enc.attention_mask,
#             "labels":         labs
#         }

#     return DataLoader(
#         ds,
#         batch_size=args.batch,
#         shuffle=shuffle,
#         collate_fn=collate,
#         pin_memory=True
#     )

# dl_train = make_loader(train_df, shuffle=True)
# dl_val   = make_loader(val_df,   shuffle=False)
# dl_test  = make_loader(test_df,  shuffle=False)

# # ─── MODEL ─────────────────────────────────────────────────────────────────────
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
#         super().__init__()
#         self.alpha, self.gamma = alpha, gamma
#         self.reduction = reduction
#     def forward(self, logits, targets):
#         bce = nn.functional.binary_cross_entropy_with_logits(
#             logits, targets, reduction="none")
#         p_t = torch.exp(-bce)
#         loss = self.alpha * (1 - p_t)**self.gamma * bce
#         return loss.mean() if self.reduction=="mean" else loss.sum()

# class ElectraMultiLabel(nn.Module):
#     def __init__(self, freeze_layers:int):
#         super().__init__()
#         self.backbone = ElectraModel.from_pretrained(
#             "google/electra-small-discriminator"
#         )
#         # freeze embeddings + bottom‐k transformer layers
#         for p in self.backbone.embeddings.parameters():
#             p.requires_grad = False
#         for i in range(freeze_layers):
#             for p in self.backbone.encoder.layer[i].parameters():
#                 p.requires_grad = False

#         # extra head
#         hidden = self.backbone.config.hidden_size
#         self.pre_classifier = nn.Sequential(
#             nn.LayerNorm(hidden),
#             nn.Dropout(0.3),
#         )
#         self.classifier = nn.Linear(hidden, N_LABELS)

#     def forward(self, input_ids, attention_mask):
#         out = self.backbone(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         ).last_hidden_state[:,0]         # [CLS]
#         h = self.pre_classifier(out)
#         return self.classifier(h)

# model = ElectraMultiLabel(args.freeze_layers).to(DEVICE)

# # ─── OPTIMIZER & SCHEDULER ─────────────────────────────────────────────────────
# # two groups: backbone vs head
# opt = AdamW([
#     {"params": [p for n,p in model.backbone.named_parameters() if p.requires_grad],
#      "lr": args.lr_backbone},
#     {"params": model.pre_classifier.parameters(),  "lr": args.lr_head},
#     {"params": model.classifier.parameters(),      "lr": args.lr_head},
# ])

# total_steps = len(dl_train) * args.epochs
# sched = OneCycleLR(
#     opt,
#     max_lr=[args.lr_backbone, args.lr_head, args.lr_head],
#     total_steps=total_steps,
#     pct_start=args.warmup_ratio,
#     anneal_strategy="cos"
# )

# criterion = FocalLoss()

# # ─── METRICS ──────────────────────────────────────────────────────────────────
# def precision_recall_f1(probs, labels, thresh=0.5):
#     p = (probs>=thresh).int()
#     y = labels.int()
#     eps=1e-9
#     tp = (p & y).sum(0).float()
#     fp = (p & ~y).sum(0).float()
#     fn = ((~p)& y).sum(0).float()
#     prec = (tp/(tp+fp+eps)).mean().item()
#     rec  = (tp/(tp+fn+eps)).mean().item()
#     f1   = 2*prec*rec/(prec+rec+eps)
#     return prec, rec, f1

# def sweep_threshold(probs, labels):
#     best = (0,0,0, -1)
#     for t in np.arange(0.0,0.55,0.05):
#         p,r,f = precision_recall_f1(probs, labels, t)
#         if f>best[3]: best=(t,p,r,f)
#     return best

# # ─── TRAIN / EVAL LOOP ────────────────────────────────────────────────────────
# def run_epoch(loader, train=False):
#     model.train() if train else model.eval()
#     total_loss, all_p, all_y = 0.0, [], []
#     torch.set_grad_enabled(train)
#     for b in loader:
#         ids   = b["input_ids"].to(DEVICE)
#         mask  = b["attention_mask"].to(DEVICE)
#         y     = b["labels"].to(DEVICE)
#         logits= model(ids,mask)
#         loss  = criterion(logits, y)
#         if train:
#             loss.backward()
#             opt.step()
#             sched.step()
#             opt.zero_grad()
#         total_loss += loss.item()
#         all_p.append(torch.sigmoid(logits).cpu())
#         all_y.append(y.cpu())
#     avg_loss = total_loss/len(loader)
#     return avg_loss, torch.cat(all_p), torch.cat(all_y)

# best_f1, best_t = -1, 0.5
# for ep in range(1, args.epochs+1):
#     tr_loss, _, _  = run_epoch(dl_train, train=True)
#     v_loss,  vp, vy= run_epoch(dl_val,   train=False)
#     t,p,r,f        = sweep_threshold(vp, vy)
#     print(f"[{ep}/{args.epochs}] train={tr_loss:.3f} val={v_loss:.3f}"
#           f" → t={t:.2f} P={p:.3f} R={r:.3f} F1={f:.3f}")
#     if f>best_f1:
#         best_f1, best_t = f,t
#         torch.save(model.state_dict(), args.out)
#         print("  ↳ saved best checkpoint")

# print(f"\n▶︎ BEST val F1={best_f1:.3f} @ t={best_t:.2f}")

# # ─── FINAL TEST ────────────────────────────────────────────────────────────────
# model.load_state_dict(torch.load(args.out))
# _, tp, ty = run_epoch(dl_test, train=False)
# p,r,f     = precision_recall_f1(tp, ty, best_t)
# print(f"[TEST] Precision={p:.3f} Recall={r:.3f} F1={f:.3f}")

#!/usr/bin/env python
"""
train_electra_base_transfer.py
Fine-tune ELECTRA-base with a small transfer head, gradient checkpointing,
differential LRs, frozen bottom layers, dynamic padding, and cosine scheduler.
"""

# import argparse, random, math, os, csv
# from collections import Counter
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import (
#     ElectraModel,
#     ElectraTokenizerFast,
#     DataCollatorWithPadding,
#     AdamW,
#     get_cosine_schedule_with_warmup
# )

# # ──────────────────────────────────────────────
# # CLI
# # ──────────────────────────────────────────────
# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--tsv",           required=True,
#                    help="tab-delimited file with columns ['plot','genres']")
#     p.add_argument("--epochs",  type=int, default=5)
#     p.add_argument("--batch",   type=int, default=4,
#                    help="per-step batch size (will accumulate)")
#     p.add_argument("--accum",   type=int, default=4,
#                    help="gradient accumulation steps")
#     p.add_argument("--lr_backbone", type=float, default=5e-6)
#     p.add_argument("--lr_head",     type=float, default=2e-5)
#     p.add_argument("--warmup_ratio",type=float, default=0.1)
#     p.add_argument("--max_len", type=int,   default=384)
#     p.add_argument("--freeze_layers", type=int, default=2,
#                    help="how many bottom encoder layers to freeze")
#     p.add_argument("--seed",    type=int, default=42)
#     p.add_argument("--out",     default="electra_base_transfer.pth")
#     return p.parse_args()

# args = parse_args()

# # ──────────────────────────────────────────────
# # Repro & Device
# # ──────────────────────────────────────────────
# torch.manual_seed(args.seed)
# random.seed(args.seed)
# np.random.seed(args.seed)

# DEVICE = torch.device("mps" if torch.backends.mps.is_available()
#                       else "cuda" if torch.cuda.is_available() else "cpu")
# print(f"→ Device: {DEVICE}")

# # ──────────────────────────────────────────────
# # Labels setup
# # ──────────────────────────────────────────────
# LABELS = ["Action","Comedy","Adventure","Thriller",
#           "Crime","Drama","Horror","Romance","Sci-Fi"]
# lab2id = {g:i for i,g in enumerate(LABELS)}
# N_LABELS = len(LABELS)

# # ──────────────────────────────────────────────
# # Data loading + split
# # ──────────────────────────────────────────────
# df = pd.read_csv(args.tsv, sep="\t", quoting=csv.QUOTE_NONE,
#                  engine="python", on_bad_lines="skip")
# print(f"→ loaded {len(df):,} rows")

# # drop any bad / empty plots
# df = df.dropna(subset=["plot","genres"]).reset_index(drop=True)

# # parse genres pipe-delimited → list
# df["tags"] = df["genres"].apply(lambda s: [g.strip() for g in s.split("|")])

# # shuffle & 80/10/10 split
# idx = np.random.permutation(len(df))
# n = len(df)
# n_train = int(0.8*n)
# n_val   = int(0.1*n)
# train_df = df.iloc[idx[:n_train]]
# val_df   = df.iloc[idx[n_train:n_train+n_val]]
# test_df  = df.iloc[idx[n_train+n_val:]]
# print(f"→ split: train {len(train_df):,},  val {len(val_df):,},  test {len(test_df):,}")

# # ──────────────────────────────────────────────
# # Tokenizer & collator
# # ──────────────────────────────────────────────
# tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")
# collator  = DataCollatorWithPadding(tokenizer, return_tensors="pt")

# # ──────────────────────────────────────────────
# # Dataset
# # ──────────────────────────────────────────────
# class MovieDS(Dataset):
#     def __init__(self, df, max_len):
#         self.plots = df["plot"].tolist()
#         self.tags  = df["tags"].tolist()
#         self.max_len = max_len

#     def __len__(self): return len(self.plots)
#     def __getitem__(self, i):
#         text = str(self.plots[i])
#         enc  = tokenizer(
#             text,
#             truncation=True,
#             padding="max_length",
#             max_length=self.max_len,
#             return_tensors="pt"
#         )
#         y = torch.zeros(N_LABELS)
#         for g in self.tags[i]:
#             if g in lab2id:
#                 y[lab2id[g]] = 1
#         item = {k:v.squeeze(0) for k,v in enc.items()}
#         item["labels"] = y
#         return item

# train_ds = MovieDS(train_df, args.max_len)
# val_ds   = MovieDS(val_df,   args.max_len)
# test_ds  = MovieDS(test_df,  args.max_len)

# dl_train = DataLoader(train_ds, batch_size=args.batch, shuffle=True, 
#                       collate_fn=collator)
# dl_val   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
#                       collate_fn=collator)
# dl_test  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
#                       collate_fn=collator)

# # ──────────────────────────────────────────────
# # Model + transfer head
# # ──────────────────────────────────────────────
# class ElectraTransferHead(nn.Module):
#     def __init__(self, freeze_layers:int):
#         super().__init__()
#         # 1) load base
#         self.backbone = ElectraModel.from_pretrained(
#             "google/electra-base-discriminator"
#         )
#         # 2) grad-checkpointing to save mem
#         self.backbone.gradient_checkpointing_enable()
#         # 3) freeze embeddings + first K encoder layers
#         for p in self.backbone.embeddings.parameters():
#             p.requires_grad = False
#         for i in range(freeze_layers):
#             for p in self.backbone.encoder.layer[i].parameters():
#                 p.requires_grad = False

#         hidden = self.backbone.config.hidden_size
#         # 4) small pre-head
#         self.pre_classifier = nn.Sequential(
#             nn.LayerNorm(hidden),
#             nn.Dropout(0.3),
#         )
#         # 5) final classification
#         self.classifier = nn.Linear(hidden, N_LABELS)

#     def forward(self, input_ids, attention_mask):
#         out = self.backbone(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#         )
#         cls = out.last_hidden_state[:,0]          # [CLS]
#         h   = self.pre_classifier(cls)
#         return self.classifier(h)

# model = ElectraTransferHead(args.freeze_layers).to(DEVICE)

# # ──────────────────────────────────────────────
# # Optimizer with differential LR
# # ──────────────────────────────────────────────
# # head params
# head_params = list(model.pre_classifier.parameters()) + \
#               list(model.classifier.parameters())
# backbone_params = [p for n,p in model.backbone.named_parameters()
#                    if p.requires_grad]

# optimizer = AdamW([
#     {"params": backbone_params, "lr": args.lr_backbone},
#     {"params": head_params,     "lr": args.lr_head},
# ], eps=1e-8)

# # ──────────────────────────────────────────────
# # Scheduler: cosine with warmup
# # ──────────────────────────────────────────────
# total_steps = len(dl_train) * args.epochs // args.accum
# warmup_steps = int(args.warmup_ratio * total_steps)
# scheduler = get_cosine_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=warmup_steps,
#     num_training_steps=total_steps
# )

# criterion = nn.BCEWithLogitsLoss()

# # ──────────────────────────────────────────────
# # Training / Eval loops
# # ──────────────────────────────────────────────
# def prf_thresh(probs, y, t):
#     p = (probs>=t).int();  Y=y.int()
#     eps=1e-9
#     tp=(p&Y).sum(0).float()
#     fp=(p&~Y).sum(0).float()
#     fn=((~p)&Y).sum(0).float()
#     prec=(tp/(tp+fp+eps)).mean().item()
#     rec =(tp/(tp+fn+eps)).mean().item()
#     f1  =2*prec*rec/(prec+rec+eps)
#     return prec,rec,f1

# def find_best_thresh(probs, y):
#     best=(0,0,0, -1)
#     for t in np.arange(0,0.55,0.05):
#         p,r,f = prf_thresh(probs,y,t)
#         if f>best[3]: best=(t,p,r,f)
#     return best

# def train_epoch():
#     model.train()
#     running_loss=0
#     optimizer.zero_grad()
#     all_p, all_y = [], []
#     for step, batch in enumerate(dl_train,1):
#         inp = batch["input_ids"].to(DEVICE)
#         att = batch["attention_mask"].to(DEVICE)
#         y   = batch["labels"].to(DEVICE)
#         logits = model(inp, att)
#         loss   = criterion(logits, y)/args.accum
#         loss.backward()
#         if step % args.accum == 0:
#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad()
#         running_loss += loss.item()*args.accum
#         all_p.append(torch.sigmoid(logits).cpu())
#         all_y.append(y.cpu())
#     return running_loss/len(dl_train), \
#            torch.cat(all_p), torch.cat(all_y)

# @torch.no_grad()
# def eval_epoch(loader):
#     model.eval()
#     total_loss=0
#     all_p, all_y = [], []
#     for batch in loader:
#         inp = batch["input_ids"].to(DEVICE)
#         att = batch["attention_mask"].to(DEVICE)
#         y   = batch["labels"].to(DEVICE)
#         logits = model(inp, att)
#         total_loss += criterion(logits, y).item()
#         all_p.append(torch.sigmoid(logits).cpu())
#         all_y.append(y.cpu())
#     return total_loss/len(loader), \
#            torch.cat(all_p), torch.cat(all_y)

# best_f1, best_t = -1, 0
# for ep in range(1, args.epochs+1):
#     tr_loss, tr_p, tr_y = train_epoch()
#     val_loss, val_p, val_y = eval_epoch(dl_val)
#     t, p, r, f1 = find_best_thresh(val_p, val_y)
#     print(f"[{ep}/{args.epochs}] train={tr_loss:.3f}  val={val_loss:.3f}  "
#           f"→ t={t:.2f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
#     if f1>best_f1:
#         best_f1, best_t = f1, t
#         torch.save(model.state_dict(), args.out)
#         print("  ↳ saved new best")

# # final test
# model.load_state_dict(torch.load(args.out))
# _, test_p, test_y = eval_epoch(dl_test)
# p,r,f1 = prf_thresh(test_p, test_y, best_t)
# print(f"\n[TEST SET] P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
