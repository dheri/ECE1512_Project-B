"""Run both notebooks on synthetic images and check their saved outputs."""

import csv
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import nbformat
from nbclient import NotebookClient
import torch
from torchmetrics.functional.image import structural_similarity_index_measure

from image_quality import rgb_ssim


ROOT = Path(__file__).resolve().parent
RESTORATION = ROOT / "jpeg_restoration_notebook"
COMPRESSION = ROOT / "dnn_compression_notebook"


def check_ssim():
    rng = torch.Generator().manual_seed(8)
    for shape in ((2, 3, 25, 31), (1, 3, 11, 11)):
        a = torch.rand(shape, generator=rng)
        b = torch.rand(shape, generator=rng)
        _, score_map = structural_similarity_index_measure(
            a, b, data_range=1.0, reduction="none", return_full_image=True
        )
        expected = score_map[..., 5:-5, 5:-5].flatten(1).mean(1)
        torch.testing.assert_close(rgb_ssim(a, b), expected, atol=1e-5, rtol=1e-4)
    print("SSIM matches the independent valid-window reference.", flush=True)


def rows_from(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def check_restoration(run):
    results = run / "results"
    rows = rows_from(results / "metrics_long.csv")
    assert len(rows) == 10
    for q in (10, 15, 20, 25, 30):
        pair = [r for r in rows if int(r["Q"]) == q]
        assert {r["Method"] for r in pair} == {"JPEG", "CNN + L1"}
        assert {int(r["Images"]) for r in pair} == {5}
        assert len({r["BPP"] for r in pair}) == 1
    assert len(list(results.glob("comparison_*.png"))) == 9
    with ZipFile(run / "jpeg_restoration_demo_results.zip") as archive:
        assert "spatial_only/best.pt" in archive.namelist()
        assert archive.read("results/metrics_long.csv") == (results / "metrics_long.csv").read_bytes()


def append_restoration_reruns(notebook):
    """Exercise retraining in the same kernel and explicit checkpoint reuse."""
    by_id = {cell.id: cell for cell in notebook.cells}
    train_and_report = [by_id[cell_id].source for cell_id in (
        "969f5c15", "95d07167", notebook.cells[30].id, notebook.cells[32].id, "ddea1c26",
    )]
    for source in train_and_report:
        notebook.cells.append(nbformat.v4.new_code_cell(source))
    notebook.cells.append(nbformat.v4.new_code_cell(
        "OUTPUT_ROOT = active_run_root\nTRAIN_MODELS = False"
    ))
    for source in train_and_report:
        notebook.cells.append(nbformat.v4.new_code_cell(source))


def check_restoration_reruns(base):
    assert list((base / "spatial_only").iterdir()) == [base / "spatial_only/config.json"]
    assert (base / "spatial_only/config.json").read_text() == "{}"
    runs = list(base.glob("run-*"))
    assert len(runs) == 2
    for run in runs:
        check_restoration(run)
    evaluation, = base.glob("run-*/evaluation-*")
    check_restoration(evaluation)
    assert not (evaluation / "spatial_only").exists()
    metrics = "results/metrics_long.csv"
    assert (evaluation / metrics).read_bytes() == (evaluation.parent / metrics).read_bytes()
    print("Restoration: interrupted output preserved, two training runs and checkpoint reuse verified.", flush=True)


def check_compression(run):
    output = next(run.glob("codec_demo_*")) / "jpeg_comparison"
    rows = rows_from(output / "per_image.csv")
    assert len(rows) == 5 * 8
    for row in rows:
        assert math.isclose(float(row["actual_bpp"]), 8 * int(row["file_bytes"]) / int(row["pixels"]))
    for i in range(4):
        row = next(r for r in rows if r["method"] == "DNN" and int(r["image"]) == i)
        assert (output / f"image_{i}.ecvq").stat().st_size == int(row["file_bytes"])
    assert (output / "decoded_image.png").is_file()


def main():
    torch.set_num_threads(min(2, torch.get_num_threads()))
    check_ssim()
    sys.path.insert(0, str(RESTORATION))
    from notebook_utils import make_demo_images

    with TemporaryDirectory(prefix="project-b-demo-") as tmp:
        data = Path(tmp) / "data"
        make_demo_images(data)
        os.environ.update(PROJECT_B_DATA_ROOT=str(data), PROJECT_B_PROFILE="demo")
        for module, name, check in (
            (RESTORATION, "JPEG_Restoration_Rerun_Safe", check_restoration_reruns),
            (COMPRESSION, "DNN_Compression_Student", check_compression),
        ):
            run = Path(tmp) / module.name
            os.environ["PROJECT_B_RUN_DIR"] = str(run)
            notebook = nbformat.read(module / f"{name}.ipynb", as_version=4)
            if module == RESTORATION:
                (run / "spatial_only").mkdir(parents=True)
                (run / "spatial_only/config.json").write_text("{}")
                append_restoration_reruns(notebook)
            nbformat.validate(notebook)
            client = NotebookClient(notebook, timeout=120, kernel_name="python3",
                                    resources={"metadata": {"path": str(module)}})
            client.execute()
            check(run)
            print(f"{name}: all cells executed and outputs verified.", flush=True)


if __name__ == "__main__":
    main()
