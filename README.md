# Multimodal Flood and Landslide Segmentation: Feasibility Study

Code and results for the paper "Multimodal Deep Learning for Flood and Landslide Segmentation: A Feasibility Study Toward a Cascading-Hazard Risk Framework" (FTNCT 2026).

## Contents
- `rerun_evaluation_colab.py`: training and evaluation script (U-Net, ResNet-34 encoder).
- `FTNCT_rerun_resumable.ipynb`: Google Colab notebook that runs the whole pipeline.
- `results_flood.json`, `results_flood_cloud.csv`: flood results (3 seeds) and cloud-occlusion results.

## Data (public, not included)
- Sen1Floods11: `gs://sen1floods11` (official hand-labelled split, Bolivia held out).
- Landslide4Sense: Zenodo record 10463239 (CC BY 4.0).

## How to run
1. Open the notebook in Google Colab with a T4 GPU.
2. Run the cells in order. Results are saved to Google Drive and finished seeds are skipped on restart.
3. Command line: `python rerun_evaluation_colab.py flood --seeds 0 1 2` or `... landslide --seeds 0 1 2`.

## Protocol
Adam, learning rate 1e-4, 30 epochs, seeds 0, 1, 2. Checkpoint chosen on validation IoU only; metrics reported on test and Bolivia chips (mean ± SD).

## Contact
[SMRUTHI.D], Alliance University, [nairPHD726@stu.alliance.edu.in]
