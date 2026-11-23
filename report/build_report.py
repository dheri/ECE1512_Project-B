"""Build the editable Project B report, its evidence bundle, and a code ZIP."""

import argparse
import base64
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
WIDTH = 516


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def stage_evidence(destination, restoration, codec, ablations):
    for source, target, entries in [
        (restoration, "restoration", ("spatial_only/best.pt", "spatial_only/config.json",
         "spatial_only/history.json", "results")),
        (codec, "codec", ("best.pt", "config.json", "data_protocol.json", "history.csv",
         "warmup.json", "test_metrics.json", "resources.json", "jpeg_comparison")),
        (ablations, "ablations", ("summary.json", "per_image.csv", "rate_distortion.png",
         "decoded_examples.png", "pruned.pt", "context.pt")),
    ]:
        if not source:
            continue
        for name in entries:
            item, out = source / name, destination / target / name
            out.parent.mkdir(parents=True, exist_ok=True)
            if item.is_dir():
                shutil.copytree(item, out, dirs_exist_ok=True)
            else:
                shutil.copyfile(item, out)


def values(evidence):
    rest = rows(evidence / "restoration/results/metrics_long.csv")
    by_q = {(int(r["Q"]), r["Method"]): r for r in rest}
    assert len(rest) == 10 and len(by_q) == 10
    table = [["Q", "File BPP", "JPEG PSNR", "CNN PSNR", "JPEG SSIM", "CNN SSIM"]]
    for q in (10, 15, 20, 25, 30):
        y, x = by_q[q, "JPEG"], by_q[q, "CNN + L1"]
        assert x["BPP"] == y["BPP"] and x["Images"] == y["Images"] == "1576"
        table.append([str(q), f"{float(y['BPP']):.3f}",
                      *(f"{float(r[key]):.{dp}f}" for key, dp in (("PSNR", 2), ("SSIM", 4))
                        for r in (y, x))])
    data = json.loads((evidence / "ablations/summary.json").read_text())
    variants = {r["method"]: r for r in data["results"] if r["q"] == ""}
    base, pruned, context = (variants[k] for k in ("VQ 256 / zlib", "VQ 128 / zlib", "VQ 256 / context"))
    ref = json.loads((evidence / "codec/jpeg_comparison/summary.json").read_text())["metrics"][0]
    assert base["bpp"] == ref["actual_bpp"] and abs(base["psnr"] - ref["mean_psnr"]) < 0.001
    assert abs(base["ssim"] - ref["mean_ssim"]) < 0.0001 and data["exact_index_and_image_checks"] == 768
    codec = [["Codec", "File BPP", "Payload BPP", "PSNR (dB)", "SSIM", "Shared KiB"]]
    for row, key in zip((base, pruned, context), ("base", "pruned", "context")):
        codec.append([row["method"], f"{row['bpp']:.3f}", f"{row['payload_bpp']:.3f}",
                      f"{row['psnr']:.2f}", f"{row['ssim']:.4f}", f"{data['checkpoints'][key]/1024:.1f}"])
    curves = {}
    for method in ("jpeg", "jpeg_opt"):
        available = [r for r in data["results"] if r["method"] == method]
        curves[method] = min(available, key=lambda r: abs(r["bpp"] - base["bpp"]))
    j, jo = curves["jpeg"], curves["jpeg_opt"]
    comparison = (f"The baseline reaches {base['psnr']:.2f} dB and {base['ssim']:.4f} at {base['bpp']:.3f} BPP. "
                  f"The nearest standard JPEG point is Q{j['q']}: {j['bpp']:.3f} BPP, {j['psnr']:.2f} dB, "
                  f"SSIM {j['ssim']:.4f}. The nearest optimized-table point is Q{jo['q']}: {jo['bpp']:.3f} BPP, "
                  f"{jo['psnr']:.2f} dB, SSIM {jo['ssim']:.4f}. These are nearby operating points, not "
                  "identical rates. The learned checkpoint is assumed to be available at the receiver.")
    active = context["modes"]["1"] + context["modes"]["2"]
    sign_word = "lowered" if context["bpp"] < base["bpp"] else "increased"
    pruning_result = (f"lowered full-file rate by {base['bpp']-pruned['bpp']:.3f} BPP"
                      if pruned["bpp"] <= base["bpp"] else
                      f"increased full-file rate by {pruned['bpp']-base['bpp']:.3f} BPP")
    discussion = (f"**What the ablations show.** The retained codewords covered {data['coverage']:.1%} of "
                  f"calibration assignments. Pruning {pruning_result} and changed PSNR by "
                  f"{pruned['psnr']-base['psnr']:+.2f} dB. Fewer nominal bits per index did not guarantee "
                  "a smaller zlib stream after reassignment and repacking. The context-enabled container chose arithmetic "
                  f"coding for {active} of 256 crops and {sign_word} full-file rate by "
                  f"{abs(context['bpp']-base['bpp']):.3f} BPP at unchanged quality. Its shared checkpoint "
                  "is larger because the decoder also needs the fitted frequency tables.")
    return {"restoration": table, "codec": codec}, comparison, discussion


def rich(line):
    runs = []
    short_links = {"doi.org": "DOI", "arxiv.org": "arXiv", "openaccess.thecvf.com": "CVF",
                   "huggingface.co": "Dataset page"}
    for part in re.split(r"(\*\*[^*]+\*\*|https?://\S+)", line):
        if part.startswith("**"):
            runs.append({"text": part[2:-2], "bold": True})
        elif part.startswith("http"):
            host = part.split("/")[2]
            runs.append({"text": short_links.get(host, part), "link": part, "decoration": "underline"})
        elif part:
            runs.append({"text": part})
    return runs


def parse(source, evidence, tables, width=WIDTH):
    nodes = []
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    iterator = iter(lines)
    for line in iterator:
        if line == "<!-- columns -->":
            left, right = [], []
            active = left
            for inner in iterator:
                if inner == "<!-- /columns -->":
                    break
                if inner == "<!-- column -->":
                    active = right
                else:
                    active.append(inner)
            column_width = (width - 17) / 2
            nodes.append({"columns": [{"width": column_width, "stack": parse("\n".join(part),
                          evidence, tables, column_width)} for part in (left, right)], "columnGap": 17})
        elif (m := re.fullmatch(r"\{\{table:(\w+)\}\}", line)):
            body = [[{"text": item, "bold": i == 0, "alignment": "center"} for item in row]
                    for i, row in enumerate(tables[m[1]])]
            widths = ["*"] * len(body[0]) if m[1] != "codec" else [99, "*", "*", "*", "*", "*"]
            nodes.append({"table": {"headerRows": 1, "widths": widths, "body": body,
                                    "dontBreakRows": True}, "layout": "lightHorizontalLines",
                          "fontSize": 8.1, "margin": [0, 1, 0, 8]})
        elif (m := re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)):
            data = base64.b64encode((evidence / m[2]).read_bytes()).decode()
            scale = .78 if m[2].endswith("decoded_examples.png") else .92
            nodes.append({"stack": [
                {"image": "data:image/png;base64," + data, "width": width*scale, "alignment": "center"},
                {"text": rich(m[1]), "fontSize": 8.1, "margin": [0, 4, 0, 9]},
            ], "unbreakable": True})
        elif line.startswith("#"):
            level, heading = line.split(" ", 1)
            nodes.append({"text": rich(heading), "style": f"h{len(level)}"})
        else:
            caption = line.startswith("Table ")
            is_ref = bool(re.match(r"^\[\d+\]", line))
            align = "center" if "University of Toronto" in line or line.startswith("Project repository:") else "justify"
            nodes.append({"text": rich(line), "alignment": align, "margin": [0, 0, 0, 4],
                          "fontSize": 8.0 if is_ref or caption else 9.4})
    return nodes


RENDER = """\
const fs = require('fs'), path = require('path');
const [lib, spec, output] = process.argv.slice(2);
const Printer = require(lib), vfs = require(path.join(lib, 'build/vfs_fonts.js'));
const font = name => Buffer.from(vfs['Roboto-' + name + '.ttf'], 'base64');
const printer = new Printer({Roboto:{normal:font('Regular'), bold:font('Medium'),
  italics:font('Italic'), bolditalics:font('MediumItalic')}});
const pdf = printer.createPdfKitDocument(JSON.parse(fs.readFileSync(spec, 'utf8')));
pdf.pipe(fs.createWriteStream(output));
pdf.end();
"""


def render(args, evidence):
    tables, comparison, discussion = values(evidence)
    source = (PROJECT / "report/report.md").read_text()
    source = (source.replace("{{repository}}", args.repo_url or "GitHub URL pending before submission")
              .replace("{{comparison}}", comparison).replace("{{ablation_discussion}}", discussion))
    pages = source.split("<!-- page -->")
    content = []
    for i, page in enumerate(pages):
        nodes = parse(page, evidence, tables)
        if i:
            nodes[0]["pageBreak"] = "before"
        content.extend(nodes)
    spec = dict(pageSize="LETTER", pageMargins=[48, 46, 48, 43],
                defaultStyle={"font": "Roboto", "fontSize": 9.4, "lineHeight": 1.12},
                styles={"h1": {"fontSize": 21, "bold": True, "alignment": "center", "margin": [0, 0, 0, 8]},
                        "h2": {"fontSize": 11.8, "bold": True, "margin": [0, 2, 0, 6]},
                        "h3": {"fontSize": 10, "bold": True, "margin": [0, 4, 0, 5]}},
                footer={"text": "ECE1512  |  Project B  |  Parteek Dheri", "fontSize": 8,
                        "alignment": "center", "color": "#575757"},
                info={"title": "JPEG Artifact Restoration and Learned Image Compression",
                      "author": "Parteek Dheri"}, content=content)
    args.work.mkdir(parents=True, exist_ok=True)
    record = args.work / "project_b.pdf.json"
    record.write_text(json.dumps(spec, ensure_ascii=False))
    pdf = args.output / "ECE1512_ProjectB_Report.pdf"
    with TemporaryDirectory(dir=args.work) as work:
        script = Path(work) / "render.cjs"
        script.write_text(RENDER)
        subprocess.run([str(args.node), str(script), str(args.pdfmake.resolve()), str(record), str(pdf)],
                       check=True)
    with pdf.open("rb") as handle:
        assert handle.read(5) == b"%PDF-"
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    assert int(re.search(r"^Pages:\s+(\d+)", info, re.M)[1]) == len(pages), info
    print(f"Verified {len(pages)}-page PDF: {pdf}")
    # Export a fully populated draft for reviewers who want the generated numbers in Markdown.
    for key, table in tables.items():
        lines = ["| " + " | ".join(row) + " |" for row in table]
        lines.insert(1, "| " + " | ".join(["---"] * len(table[0])) + " |")
        source = source.replace(f"{{{{table:{key}}}}}", "\n".join(lines))
    source = re.sub(r"^(!\[.*?\]\()(restoration|ablations)/", r"\g<1>evidence/\2/", source, flags=re.M)
    (args.output / "ECE1512_ProjectB_Editable.md").write_text(source)


def package(args, evidence):
    dest = args.output / "ECE1512_ProjectB_Code.zip"
    output = args.output.resolve()
    nested_output = PROJECT in output.parents
    editable = "ECE1512_ProjectB_Editable.md"
    pdf = "ECE1512_ProjectB_Report.pdf"
    notebooks = {"JPEG_Restoration_Rerun_Safe.ipynb", "DNN_Compression_Student.ipynb"}
    with ZipFile(dest, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(PROJECT.rglob("*")):
            rel = path.relative_to(PROJECT)
            if rel == Path(editable) or (nested_output and path.is_relative_to(output)):
                continue
            if any(part.startswith(".") or part in ("runs", "data_1um_bicubic", "__pycache__", "evidence")
                   for part in rel.parts):
                continue
            if not path.is_file() or not (path.suffix in (".py", ".md", ".txt") or path.name in notebooks):
                continue
            if path.name in notebooks:
                notebook = json.loads(path.read_text())
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        cell["outputs"], cell["execution_count"] = [], None
                    cell.get("metadata", {}).pop("execution", None)
                notebook.get("metadata", {}).pop("widgets", None)
                archive.writestr(str(rel), json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
            else:
                archive.write(path, rel)
        for path in sorted(evidence.rglob("*")):
            if path.is_file():
                archive.write(path, Path("evidence") / path.relative_to(evidence))
        archive.write(args.output / editable, editable)
        archive.write(args.output / pdf, Path("report") / pdf)
    with ZipFile(dest) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), "Duplicate paths in the code ZIP"
        assert archive.testzip() is None
        assert {editable, f"report/{pdf}", "report/report.md", "evidence/codec/best.pt",
                "evidence/ablations/per_image.csv", "evidence/restoration/results/metrics_long.csv"} <= set(names)
        print(f"Verified complete code ZIP ({len(names)} files): {dest}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node", type=Path, default=ROOT/"tmp/pdf_runtime/nodejs_wheel/bin/node")
    p.add_argument("--pdfmake", type=Path, default=ROOT/"tmp/pdf_runtime/node_modules/pdfmake")
    p.add_argument("--work", type=Path, default=ROOT/"tmp")
    p.add_argument("--output", type=Path, default=ROOT/"outputs")
    p.add_argument("--repo-url", default="")
    p.add_argument("--restoration-run", type=Path)
    p.add_argument("--codec-run", type=Path)
    p.add_argument("--ablations-run", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = (args.restoration_run, args.codec_run, args.ablations_run)
    if any(sources) and not all(sources):
        p.error("when collecting evidence, pass all three source directories")
    evidence = args.output / "evidence" if all(sources) else PROJECT / "evidence"
    if all(sources):
        stage_evidence(evidence, *sources)
    render(args, evidence)
    package(args, evidence)


if __name__ == "__main__":
    main()
