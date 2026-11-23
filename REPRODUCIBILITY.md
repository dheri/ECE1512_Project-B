# Project B reproducibility checklist

- **Data:** Use the prepared `train/`, `val/`, and `test/` directories from [ADPPatches](https://huggingface.co/datasets/LeonYe/ADPPatches), `data_1um_bicubic.zip`, SHA-256 `219249b42106a7566711e8b4401018d0db61ca27743924093115382e44a670cd`. The archive contains 12,616 / 1,576 / 1,576 PNGs. It is not copied into the code ZIP. Prepared splits overlap by source filename; this is a patch-level holdout.
- **Restoration:** Use `jpeg_restoration_notebook/JPEG_Restoration_Rerun_Safe.ipynb`, profile `full`, seed 42. The full configuration, training history, best checkpoint, all required test qualities, and figures are in `evidence/restoration/`.
- **Learned compression:** Use `dnn_compression_notebook/DNN_Compression_Student.ipynb`, profile `full`, seed 1512. `evidence/codec/data_protocol.json` records the exact ordered relative paths (1,024 training, 128 validation, 256 test). The archived JSON absolute paths are run provenance, not reproduction requirements. For a new location, let the notebook create a fresh split manifest.
- **Hyperparameters and model selection:** Use the supplied configuration JSONs. Checkpoint selection uses restored validation PSNR for restoration and the hard rate–distortion objective for the codec. `evidence/codec/config.json` is authoritative for the actual quantized temperature schedule; `warmup.json` records the initialization routine's suggested endpoint, which the notebook replaced.
- **Quality and rate:** Use normalized RGB, per-image PSNR and valid-window Gaussian SSIM. JPEG uses Pillow with 4:4:4 sampling. Compare file rates with headers included; shared neural weights are disclosed separately. Payload-only and estimated rates are diagnostics.
- **Ablations:** Both use the same saved baseline checkpoint and feed only the original training paths into codeword frequency estimation and context-table fitting. No new gradient training is performed. The code verifies exact recovered indices and images for three learned variants across 256 crops. The decoded JPEG reference is also verified unchanged by Huffman table optimization.
- **Environment:** The completed experiments ran on CPU, Python 3.12, PyTorch 2.14.1. Install `requirements.txt` in a suitable environment. `python check_baselines.py` runs the synthetic checks without the dataset.
- **Report:** `report/report.md` is the editable build source. Figure and table inputs are in `evidence/` in the extracted code package. The filled Markdown draft is also included.

## Reproduce the restoration results

From the repository root with the prepared dataset under `data_1um_bicubic/`, run:

```bash
python jpeg_restoration_notebook/rerun_evaluation.py --output reproduced_restoration
```

This runs Q10, Q15, Q20, Q25, and Q30 on CPU using the bundled checkpoint, without retraining. Compare `reproduced_restoration/metrics_long.csv` against `evidence/restoration/results/metrics_long.csv`. The script uses the notebook's valid-window SSIM. Pass `--data-root /path/to/data_1um_bicubic` if the dataset is elsewhere and choose a new output directory for each run.

## Reproduce the codec comparison and ablations

From the extracted code package root, with the dataset under `data_1um_bicubic/` (or an alternate `--data-root`):

```bash
python dnn_compression_notebook/evaluate_ablations.py \
  --checkpoint evidence/codec/best.pt \
  --protocol evidence/codec/data_protocol.json \
  --data-root data_1um_bicubic \
  --output reproduced_ablations
```

The output path must be new. Use the resulting `summary.json`, `per_image.csv`, and PNGs to check the reported comparison. No dataset or checkpoint is uploaded.

## Rebuild the report

Install Node and `pdfmake@0.2.20` in an isolated prefix, and install Poppler's `pdfinfo`. Run this from the extracted package:

```bash
python report/build_report.py \
  --node "$(command -v node)" \
  --pdfmake /path/to/node_modules/pdfmake \
  --output "$PWD/rebuilt"
```

The report source already contains the submission repository URL. Use the bundled checkpoints for exact comparisons; use the full notebook profiles only to retrain.
