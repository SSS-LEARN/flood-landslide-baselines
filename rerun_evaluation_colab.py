"""
Re-run evaluation for the FTNCT paper (Sen1Floods11 + Landslide4Sense) - v2.

Changes in v2 (so a Colab disconnect does not lose work):
  * Every finished model is saved to RESULT_DIR (put this on Google Drive) and is
    skipped on the next run, so you can just re-run the same command to resume.
  * Chips are read from disk once and kept in memory (float16), which makes
    training much faster than re-reading TIFF files every epoch.
  * Progress is printed every epoch.

Usage:
  RESULT_DIR=/content/drive/MyDrive/ftnct_rerun/results python rerun_evaluation_colab.py flood --seeds 0 1 2
  RESULT_DIR=... python rerun_evaluation_colab.py landslide --seeds 0 1 2
  python rerun_evaluation_colab.py smoke
"""
import argparse, csv, glob, json, os, random, sys, tempfile, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLOOD_ROOT = os.environ.get("FLOOD_ROOT", "data/HandLabeled")
SPLIT_DIR = os.environ.get("SPLIT_DIR", "splits")
L4S_ROOT = os.environ.get("L4S_ROOT", "data/L4S")
RESULT_DIR = os.environ.get("RESULT_DIR", "results")
EPOCHS = int(os.environ.get("EPOCHS", 30))
LR = 1e-4
OCCLUSION_LEVELS = [0, 5, 10, 15, 20, 30, 40, 60, 80, 100]
N_DRAWS = 5
PRETRAINED = "imagenet"


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class Conf:
    """Accumulates TP/FP/FN over a split, ignoring label 255."""
    def __init__(self): self.tp = self.fp = self.fn = 0
    def update(self, logits, y):
        pred = (logits.squeeze(1) > 0).long()
        valid = y != 255
        p, t = pred[valid], y[valid]
        self.tp += int(((p == 1) & (t == 1)).sum())
        self.fp += int(((p == 1) & (t == 0)).sum())
        self.fn += int(((p == 0) & (t == 1)).sum())
    def result(self):
        tp, fp, fn = self.tp, self.fp, self.fn
        iou = tp / max(tp + fp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return dict(iou=iou, f1=f1, precision=prec, recall=rec)


def masked_bce(logits, y):
    valid = y != 255
    l = nn.functional.binary_cross_entropy_with_logits(
        logits.squeeze(1), y.float(), reduction="none")
    return l[valid].mean()


def dice_bce(logits, y):
    valid = y != 255
    p = torch.sigmoid(logits.squeeze(1))
    t = y.float()
    inter = (p * t * valid).sum()
    dice = 1 - (2 * inter + 1) / ((p * valid).sum() + (t * valid).sum() + 1)
    return masked_bce(logits, y) + dice


def evaluate(model, loader, occlude=None, rng=None):
    model.eval(); c = Conf()
    with torch.no_grad():
        for x, y in loader:
            if occlude is not None:
                x = occlude_batch(x, occlude, rng)
            c.update(model(x.to(DEV)).cpu(), y)
    return c.result()


def occlude_batch(x, frac, rng):
    x = x.clone()
    if frac <= 0: return x
    H, W = x.shape[-2:]
    for i in range(x.shape[0]):
        if frac >= 100:
            x[i, 2:] = 0; continue
        area = frac / 100 * H * W
        h = int(min(H, max(1, round((area * H / W) ** 0.5))))
        w = int(min(W, max(1, round(area / h))))
        r0 = int(rng.integers(0, H - h + 1)); c0 = int(rng.integers(0, W - w + 1))
        x[i, 2:, r0:r0 + h, c0:c0 + w] = 0
    return x


def fit(model, train_dl, val_dl, loss_fn, ckpt):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best = -1; t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_dl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad(); loss_fn(model(x), y).backward(); opt.step()
        v = evaluate(model, val_dl)["iou"]
        if v > best:
            best = v; torch.save(model.state_dict(), ckpt)
        print(f"  epoch {ep + 1}/{EPOCHS} val_iou={v:.4f} best={best:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(torch.load(ckpt, map_location=DEV))
    return best


def mean_sd(rows, keys=("iou", "f1", "precision", "recall")):
    return {k: [float(np.mean([r[k] for r in rows])),
                float(np.std([r[k] for r in rows]))] for k in keys}


class FloodDS(Dataset):
    def __init__(self, names, use_optical):
        self.items, self.opt = names, use_optical
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        import rasterio
        s1n, lbn = self.items[i]
        with rasterio.open(os.path.join(FLOOD_ROOT, "S1Hand", s1n)) as f:
            s1 = np.nan_to_num(f.read().astype(np.float32), nan=0.0)
        s1 = (np.clip(s1, -50, 1) + 50) / 51.0
        x = [s1]
        if self.opt:
            with rasterio.open(os.path.join(
                    FLOOD_ROOT, "S2Hand", s1n.replace("S1Hand", "S2Hand"))) as f:
                s2 = np.nan_to_num(f.read().astype(np.float32), nan=0.0)
            x.append(np.clip(s2, 0, 10000) / 10000.0)
        with rasterio.open(os.path.join(FLOOD_ROOT, "LabelHand", lbn)) as f:
            y = f.read(1).astype(np.int64)
        y[y == -1] = 255
        return torch.from_numpy(np.concatenate(x)), torch.from_numpy(y)


class ListDS(Dataset):
    """Chips held in memory as float16 images and uint8 labels (255 = ignore)."""
    def __init__(self, xs, ys): self.xs, self.ys = xs, ys
    def __len__(self): return len(self.xs)
    def __getitem__(self, i): return self.xs[i].float(), self.ys[i].long()


PRE = {}


def load_split(name, opt):
    key = (name, opt)
    if key not in PRE:
        ds = FloodDS(read_split(name), opt)
        xs, ys = [], []
        for i in range(len(ds)):
            x, y = ds[i]; xs.append(x.half()); ys.append(y.to(torch.uint8))
        PRE[key] = ListDS(xs, ys)
        print("cached", name, "optical" if opt else "sar", len(xs), "chips", flush=True)
    return PRE[key]


def read_split(name):
    with open(os.path.join(SPLIT_DIR, f"flood_{name}_data.csv")) as f:
        return [(r[0].strip(), r[1].strip()) for r in csv.reader(f) if r]


def run_flood(seeds, loaders=None):
    os.makedirs(RESULT_DIR, exist_ok=True)
    out = {"sar": {"test": [], "bolivia": []}, "fusion": {"test": [], "bolivia": []}}
    cloud_rows, sar_test_ious = [], []
    for mode, opt in (("sar", False), ("fusion", True)):
        for s in seeds:
            path = os.path.join(RESULT_DIR, f"flood_{mode}_s{s}.json")
            if os.path.exists(path):
                rec = json.load(open(path)); print("skip (already done):", path, flush=True)
            else:
                seed_all(s)
                if loaders:
                    tr, va, te, bo = loaders(opt)
                else:
                    mk = lambda n, sh: DataLoader(load_split(n, opt), batch_size=8,
                                                  shuffle=sh, num_workers=0)
                    tr, va, te, bo = mk("train", True), mk("valid", False), mk("test", False), mk("bolivia", False)
                model = smp.Unet("resnet34", encoder_weights=PRETRAINED,
                                 in_channels=15 if opt else 2, classes=1).to(DEV)
                print(f"== training {mode} seed {s}", flush=True)
                fit(model, tr, va, masked_bce, f"flood_{mode}_s{s}.pt")
                rec = dict(test=evaluate(model, te), bolivia=evaluate(model, bo), cloud=[])
                print(mode, "seed", s, "TEST", rec["test"], "BOLIVIA", rec["bolivia"], flush=True)
                if mode == "fusion":
                    te_cached = list(te)          # read once, reuse for every occlusion draw
                    for lvl in OCCLUSION_LEVELS:
                        for d in range(N_DRAWS):
                            rng = np.random.default_rng(1000 * s + 10 * d + lvl)
                            rec["cloud"].append(dict(seed=s, level=lvl, draw=d,
                                                     **evaluate(model, te_cached, lvl, rng)))
                    print("cloud sweep done for seed", s, flush=True)
                json.dump(rec, open(path, "w"))
            out[mode]["test"].append(rec["test"]); out[mode]["bolivia"].append(rec["bolivia"])
            if mode == "sar": sar_test_ious.append(rec["test"]["iou"])
            else: cloud_rows.extend(rec["cloud"])
    summary = {m: {sp: mean_sd(out[m][sp]) for sp in out[m]} for m in out}
    sar_mean = float(np.mean(sar_test_ious))
    by_lvl = {l: float(np.mean([r["iou"] for r in cloud_rows if r["level"] == l]))
              for l in OCCLUSION_LEVELS}
    cross = None
    lv = OCCLUSION_LEVELS
    for a, b in zip(lv[:-1], lv[1:]):
        if by_lvl[a] >= sar_mean > by_lvl[b]:
            cross = a + (by_lvl[a] - sar_mean) / (by_lvl[a] - by_lvl[b]) * (b - a); break
    summary["cloud"] = dict(sar_only_test_iou_mean=sar_mean, fusion_iou_by_level=by_lvl,
                            crossover_percent=cross)
    json.dump(dict(summary=summary, raw=out), open(os.path.join(RESULT_DIR, "results_flood.json"), "w"), indent=2)
    with open(os.path.join(RESULT_DIR, "results_flood_cloud.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cloud_rows[0])); w.writeheader(); w.writerows(cloud_rows)
    print(json.dumps(summary, indent=2))
    print("FINAL FILES in", RESULT_DIR, ": results_flood.json, results_flood_cloud.csv")
    return summary


L4S_CACHE = {}
def l4s_load(cid):
    if cid not in L4S_CACHE:
        import h5py
        with h5py.File(os.path.join(L4S_ROOT, "img", f"image_{cid}.h5"), "r") as f:
            x = np.nan_to_num(np.asarray(f["img"], dtype=np.float32)).transpose(2, 0, 1).astype(np.float16)
        with h5py.File(os.path.join(L4S_ROOT, "mask", f"mask_{cid}.h5"), "r") as f:
            y = np.asarray(f["mask"], dtype=np.uint8)
        L4S_CACHE[cid] = (x, y)
    return L4S_CACHE[cid]


class L4SDS(Dataset):
    def __init__(self, ids, augment):
        self.ids, self.aug = ids, augment
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        x, y = l4s_load(self.ids[i])
        x = torch.from_numpy(x.astype(np.float32)); y = torch.from_numpy(y.astype(np.int64))
        if self.aug:
            if random.random() < .5: x, y = x.flip(-1), y.flip(-1)
            if random.random() < .5: x, y = x.flip(-2), y.flip(-2)
        return x, y


def run_landslide(seeds, ids=None, in_ch=14):
    os.makedirs(RESULT_DIR, exist_ok=True)
    if ids is None:
        ids = sorted(os.path.basename(p)[6:-3] for p in glob.glob(os.path.join(L4S_ROOT, "img", "image_*.h5")))
    print("caching all chips in memory (one-time, a few minutes)...", flush=True)
    for k, cid in enumerate(ids):
        l4s_load(cid)
        if k % 500 == 0: print("  cached", k, "/", len(ids), flush=True)
    rows = []
    for s in seeds:
        path = os.path.join(RESULT_DIR, f"landslide_s{s}.json")
        if os.path.exists(path):
            rows.append(json.load(open(path))); print("skip (already done):", path, flush=True); continue
        seed_all(s)
        perm = np.random.default_rng(s).permutation(len(ids))
        n1, n2 = int(.70 * len(ids)), int(.85 * len(ids))
        sp = lambda a: [ids[i] for i in perm[a]]
        tr_ids, va_ids, te_ids = sp(slice(0, n1)), sp(slice(n1, n2)), sp(slice(n2, None))
        dl = lambda i, a, sh: DataLoader(L4SDS(i, a), batch_size=16, shuffle=sh, num_workers=0)
        model = smp.Unet("resnet34", encoder_weights=PRETRAINED, in_channels=in_ch, classes=1).to(DEV)
        print(f"== training landslide seed {s}", flush=True)
        fit(model, dl(tr_ids, True, True), dl(va_ids, False, False), dice_bce, f"landslide_s{s}.pt")
        r = evaluate(model, dl(te_ids, False, False)); rows.append(r)
        json.dump(r, open(path, "w"))
        print("landslide seed", s, "TEST", r, flush=True)
    summ = mean_sd(rows)
    json.dump(dict(summary=summ, raw=rows), open(os.path.join(RESULT_DIR, "results_landslide.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))
    print("FINAL FILE in", RESULT_DIR, ": results_landslide.json")
    return summ


def smoke():
    """Tiny synthetic run to prove the code executes end to end (no real data)."""
    global EPOCHS, PRETRAINED, OCCLUSION_LEVELS, N_DRAWS, RESULT_DIR
    EPOCHS, PRETRAINED, OCCLUSION_LEVELS, N_DRAWS = 1, None, [0, 50, 100], 1
    RESULT_DIR = tempfile.mkdtemp()
    def fake(n, ch):
        x = torch.rand(n, ch, 64, 64); y = (x[:, 0] > .7).long(); y[:, :2] = 255
        return DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=4)
    loaders = lambda opt: tuple(fake(8, 15 if opt else 2) for _ in range(4))
    run_flood([0], loaders)
    print("flood smoke OK")
    run_flood([0], loaders)
    print("resume smoke OK (second call skipped finished models)")
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=14, classes=1)
    x = torch.rand(4, 14, 128, 128); y = (torch.rand(4, 128, 128) > .9).long()
    l = dice_bce(m(x), y); l.backward(); print("landslide loss OK", float(l.detach()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["flood", "landslide", "smoke"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    if a.task == "flood": run_flood(a.seeds)
    elif a.task == "landslide": run_landslide(a.seeds)
    else: smoke()
