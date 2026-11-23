# ECE1512 Project B: Image Restoration and Learned Compression

**Parteek Dheri · University of Toronto · Fall 2026**

This repository contains the notebooks, checkpoints, and results for the JPEG restoration and learned compression experiments, including the codec ablations described in the [report](report/ECE1512_ProjectB_Report.pdf).

## Start here

- [Report source](report/report.md) and [populated Markdown draft](ECE1512_ProjectB_Editable.md)
- [Reproducibility checklist and commands](REPRODUCIBILITY.md)
- [JPEG restoration notebook](jpeg_restoration_notebook/JPEG_Restoration_Rerun_Safe.ipynb) and [saved test metrics](evidence/restoration/results/metrics_long.csv)
- [Learned compression notebook](dnn_compression_notebook/DNN_Compression_Student.ipynb) and [saved codec results](evidence/codec/jpeg_comparison/summary.json)
- [Ablation script](dnn_compression_notebook/evaluate_ablations.py), [results](evidence/ablations/summary.json), and [rate–distortion plot](evidence/ablations/rate_distortion.png)

## Main results

- On 1,576 test crops, residual restoration improved mean PSNR by **1.88–2.03 dB** across JPEG Q10, Q15, Q20, Q25, and Q30, at the same input bitrate.
- On 256 test crops, the learned codec reached **31.51 dB PSNR**, **0.9134 SSIM**, and **1.596 full-file bits/pixel**. Context-based coding reduced rate to **1.451 bits/pixel** with identical decoded pixels. Post-training codebook pruning to 128 entries increased rate to **1.825 bits/pixel** and reduced PSNR to **30.90 dB**.
- These experiments hold out patches, not all source images. The learned rates include file headers but do not include the shared checkpoint, which the report accounts for separately.

## Run locally

1. Extract the prepared `train/`, `val/`, and `test/` folders from [ADPPatches](https://huggingface.co/datasets/LeonYe/ADPPatches) into `data_1um_bicubic/` at the repository root, or set `PROJECT_B_DATA_ROOT`. The dataset hash and split counts are in the checklist; no raw data is bundled.
2. Install `requirements.txt` in an environment with a suitable PyTorch build and start JupyterLab. Open each notebook from its own directory with a fresh kernel. The default `full` profile writes a new run folder. Set `PROJECT_B_PROFILE=pilot` before starting the kernel for a smaller run.
3. Run `python check_baselines.py` for a synthetic end-to-end check without the dataset. To reproduce both experiments from the saved checkpoints without retraining, see the [reproducibility checklist](REPRODUCIBILITY.md).

The course starter and assignment are credited in the report.
