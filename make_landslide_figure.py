"""Landslide figure: input RGB | slope | ground truth | model prediction (test chips, seed 0).
Needs /content/landslide_s0.pt, the data link /content/data/L4S, and rerun_evaluation_colab.py in /content.
Run:  python make_landslide_figure.py
"""
import os, sys, glob
os.environ.setdefault("L4S_ROOT", "/content/data/L4S")
sys.path.insert(0, "/content")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import rerun_evaluation_colab as R

SEED = 0
CKPT = os.environ.get("CKPT", "/content/landslide_s0.pt")
OUT = os.environ.get("OUT", "/content/drive/MyDrive/ftnct_rerun/results/fig_landslide_predictions.png")
N = 4

ids = sorted(os.path.basename(p)[6:-3] for p in glob.glob(os.path.join(R.L4S_ROOT, "img", "image_*.h5")))
perm = np.random.default_rng(SEED).permutation(len(ids))          # same split as the training run
n2 = int(.85 * len(ids))
test_ids = [ids[i] for i in perm[n2:]]
m = smp.Unet("resnet34", encoder_weights=None, in_channels=14, classes=1).to(R.DEV)
m.load_state_dict(torch.load(CKPT, map_location=R.DEV)); m.eval()

# choose test chips with a good amount of landslide so the figure is informative
area = [int((R.l4s_load(c)[1] == 1).sum()) for c in test_ids]
order = np.argsort(area)[::-1]
pick = [test_ids[int(order[k])] for k in np.linspace(0, min(60, len(order) - 1), N).astype(int)]

fig, ax = plt.subplots(N, 4, figsize=(8, 2.1 * N), dpi=300)
titles = ["Sentinel-2 RGB", "Slope", "Ground truth", "Prediction"]
for r, c in enumerate(pick):
    x, y = R.l4s_load(c)
    xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(R.DEV)
    with torch.no_grad():
        pred = (m(xt).squeeze() > 0).cpu().numpy()
    xf = x.astype(np.float32)
    rgb = xf[[3, 2, 1]].transpose(1, 2, 0)                       # B4, B3, B2 (channels 3,2,1)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    panels = [rgb, xf[12], y.astype(float), pred.astype(float)]   # channel 12 = slope
    for k, im in enumerate(panels):
        ax[r, k].imshow(im, cmap=None if k == 0 else ("magma" if k == 1 else "Reds"),
                        vmin=None if k < 2 else 0, vmax=None if k < 2 else 1)
        ax[r, k].axis("off")
        if r == 0: ax[r, k].set_title(titles[k], fontsize=8)
plt.tight_layout(); os.makedirs(os.path.dirname(OUT), exist_ok=True); plt.savefig(OUT); print("saved", OUT)
