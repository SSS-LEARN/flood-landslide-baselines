# SAR–Optical Deep Learning for Flood and Landslide Segmentation

Code and results for the paper "Multimodal SAR–Optical Deep Learning for Flood and Landslide Segmentation: Baselines and Cloud-Cover Robustness" (FTNCT 2026).

## Files
| File | Purpose |
|---|---|
| `rerun_evaluation_colab.py` | Training and evaluation script (U-Net, ResNet-34 encoder). |
| `FTNCT_rerun_resumable.ipynb` | Google Colab notebook that runs the whole pipeline. |
| `results_flood.json` | Flood results (SAR-only and SAR + optical), 3 seeds. |
| `results_flood_cloud.csv` | Cloud-occlusion results for the fused flood model. |
| `results_landslide.json` | Landslide results, 3 seeds. |
| `make_landslide_figure.py` | Draws the landslide ground-truth vs prediction figure. |
| `fig_landslide_predictions.png` | The figure used in the paper. |

## Data (public, not included)
- Sen1Floods11: `gs://sen1floods11` (official hand-labelled split; Bolivia held out).
- Landslide4Sense: Zenodo record 10463239 (CC BY 4.0), file `TrainData.zip`.

## Results (mean ± SD over 3 seeds, test set)
| Model | IoU | F1 |
|---|---|---|
| SAR-only flood | 0.672 ± 0.009 | 0.804 ± 0.006 |
| SAR + optical flood | 0.822 ± 0.002 | 0.902 ± 0.001 |
| Landslide (optical + slope + DEM) | 0.587 ± 0.011 | 0.740 ± 0.008 |

Held-out Bolivia region (flood): 0.660 (SAR-only) and 0.773 (fused). The fusion advantage falls to the SAR-only level at about 9% simulated cloud on average (about 6%, 7% and 43% for the three seeds).

## How to run (Google Colab, T4 GPU)
1. Open `FTNCT_rerun_resumable.ipynb` in Colab and run the cells in order, or run the script directly:
   - `python rerun_evaluation_colab.py flood --seeds 0 1 2`
   - `python rerun_evaluation_colab.py landslide --seeds 0 1 2`
2. Set `RESULT_DIR` to a folder on Google Drive. Finished seeds are saved there and skipped when you rerun.

## Protocol
Adam, learning rate 1e-4, 30 epochs, batch size 8 (flood) and 16 (landslide), seeds 0, 1 and 2. The best checkpoint is chosen on validation IoU only, and scores are reported on test and Bolivia chips. Flood models share the official split. Each landslide seed uses its own random 70/15/15 split of the 3,799 labelled training chips. Cloud occlusion sets a random rectangle of the optical channels to zero (0 to 100% of the chip area).

## Limitations
Three seeds only; simulated (not real) cloud; the landslide split is random because the dataset does not expose the source area of each chip.

## Contact
Smruthi D, Alliance University
