"""
Download and preprocess VinDr-CXR from PhysioNet for GRAPE N10 evaluation.

VinDr-CXR (PhysioNet doi:10.13026/x4k8-8538):
  - 18,000 chest X-rays (15,000 train + 3,000 test)
  - DICOM format, ~40 GB total
  - 14 radiological findings annotated by 3 radiologists each
  - Bounding box annotations for findings in the training set

Prerequisites:
  1. Create a PhysioNet account at https://physionet.org/register/
  2. Complete the VinDr-CXR Data Use Agreement at:
     https://physionet.org/content/vindr-cxr/1.0.0/
  3. Set credentials:
       export PHYSIONET_USER=your_username
       export PHYSIONET_PASS=your_password
  4. Run:
       python scripts/download_vindrcxr.py

Output layout (matches CSRDataset expectations):
  /ephemeral/data/vindrcxr/
    images/           ← PNG files converted from DICOM
    labels.csv        ← image_id, class_label, split, concept_0..concept_13
    bboxes.csv        ← image_id, concept_idx, x1, y1, x2, y2

Splits:
  train:     70% of all images
  val:       15% of all images
  test:      15% of all images (no bboxes, pure classification eval)
  bbox_eval: up to 500 images from train split that have ≥1 bbox annotation
             (used for Pointing Game evaluation)

Consensus rule (3 radiologists per image):
  Finding present if ≥ 2/3 radiologists labeled it.
  Bbox: union of all radiologist boxes for that finding.

Estimated runtime: 2–4 hours (download-limited; parallelised conversion).
"""

import os
import sys
import csv
import json
import random
import argparse
import subprocess
import concurrent.futures
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────

PHYSIONET_BASE = "https://physionet.org/files/vindr-cxr/1.0.0"
OUTPUT_DIR     = "/ephemeral/data/vindrcxr"
DICOM_DIR      = "/ephemeral/data/vindrcxr_raw"   # temporary DICOM store
MAX_WORKERS    = 8       # parallel DICOM conversion workers
SEED           = 42

# Number of finding-level radiologist votes needed to call a concept present
MIN_VOTES      = 2       # ≥2 of 3 → consensus positive

# How many bbox_eval images to carve out (images with ≥1 annotated finding)
MAX_BBOX_EVAL  = 500

# Splits (sum = 1.0)
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

# Mapping: VinDr-CXR finding name → CONCEPTS["vindrcxr"] index
FINDING_TO_IDX = {
    "Aortic enlargement": 0,
    "Atelectasis":        1,
    "Calcification":      2,
    "Cardiomegaly":       3,
    "Consolidation":      4,
    "ILD":                5,
    "Infiltration":       6,
    "Lung Opacity":       7,
    "Nodule/Mass":        8,
    "Other lesion":       9,
    "Pleural effusion":   10,
    "Pleural thickening": 11,
    "Pneumothorax":       12,
    "Pulmonary fibrosis": 13,
}
NUM_CONCEPTS = 14
NO_FINDING_LABEL = "No finding"


# ── Download helpers ──────────────────────────────────────────────────────────

def wget_file(url: str, dest: str, user: str, password: str) -> bool:
    """Download a single file using wget with PhysioNet credentials."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        return True
    cmd = [
        "wget", "-q", "-O", dest,
        "--user", user, "--password", password,
        url,
    ]
    ret = subprocess.run(cmd, capture_output=True)
    if ret.returncode != 0 or os.path.getsize(dest) < 100:
        os.remove(dest) if os.path.exists(dest) else None
        return False
    return True


def download_annotations(user: str, password: str) -> dict:
    """Download and return paths to annotation CSVs."""
    ann_dir = os.path.join(DICOM_DIR, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    files = {
        "image_labels_train": "annotations/image_labels_train.csv",
        "image_labels_test":  "annotations/image_labels_test.csv",
        "train_bboxes":       "annotations/train.csv",
    }
    paths = {}
    for key, rel_path in files.items():
        dest = os.path.join(DICOM_DIR, rel_path)
        url  = f"{PHYSIONET_BASE}/{rel_path}"
        print(f"  Downloading {rel_path} ...", end=" ", flush=True)
        ok = wget_file(url, dest, user, password)
        print("OK" if ok else "FAILED")
        if not ok:
            raise RuntimeError(f"Failed to download {rel_path}. Check credentials and DUA.")
        paths[key] = dest
    return paths


def download_dicoms_batch(image_ids: list, split_dir: str,
                          user: str, password: str) -> int:
    """Download a batch of DICOM files. Returns number successfully downloaded."""
    ok = 0
    for iid in image_ids:
        dest = os.path.join(DICOM_DIR, split_dir, iid + ".dicom")
        url  = f"{PHYSIONET_BASE}/{split_dir}/{iid}.dicom"
        if wget_file(url, dest, user, password):
            ok += 1
    return ok


# ── Annotation parsing ────────────────────────────────────────────────────────

def parse_global_labels(labels_csv: str) -> dict:
    """
    Parse image_labels_{train,test}.csv.

    Expected columns: image_id, rad_id, <finding_name>, ..., No finding
    Returns: image_id → {finding_name → vote_count (int)}
    """
    votes = defaultdict(lambda: defaultdict(int))
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        # Finding columns: everything except image_id and rad_id
        finding_cols = [h for h in headers if h not in ("image_id", "rad_id")]
        for row in reader:
            iid = row["image_id"]
            for col in finding_cols:
                val = row[col].strip()
                if val == "1" or val.lower() == "true":
                    votes[iid][col] += 1
    return dict(votes)


def parse_bbox_annotations(train_csv: str) -> dict:
    """
    Parse train.csv (local bounding box annotations).

    Expected columns: image_id, class_name, x_min, y_min, x_max, y_max, rad_id
    Returns: image_id → {finding_name → list of (x1, y1, x2, y2, rad_id)}
    """
    bboxes = defaultdict(lambda: defaultdict(list))
    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cname = row.get("class_name", row.get("class_name", "")).strip()
            if cname == NO_FINDING_LABEL or cname not in FINDING_TO_IDX:
                continue
            try:
                x1 = int(float(row["x_min"]))
                y1 = int(float(row["y_min"]))
                x2 = int(float(row["x_max"]))
                y2 = int(float(row["y_max"]))
                rad = row.get("rad_id", "r1")
            except (ValueError, KeyError):
                continue
            if x2 > x1 and y2 > y1:
                bboxes[row["image_id"]][cname].append((x1, y1, x2, y2, rad))
    return dict(bboxes)


def consensus_labels(votes: dict, min_votes: int = MIN_VOTES) -> dict:
    """
    Apply consensus rule: finding is positive if ≥ min_votes radiologists voted it.

    Returns: image_id → {
        "concepts": [0/1 × NUM_CONCEPTS],
        "class_label": 0 (no finding) or 1 (any finding)
    }
    """
    result = {}
    for iid, vote_dict in votes.items():
        concepts = [0] * NUM_CONCEPTS
        for finding, idx in FINDING_TO_IDX.items():
            if vote_dict.get(finding, 0) >= min_votes:
                concepts[idx] = 1
        class_label = 1 if any(concepts) else 0
        result[iid] = {"concepts": concepts, "class_label": class_label}
    return result


def consensus_bboxes(bbox_dict: dict) -> dict:
    """
    Consensus bboxes: take the union (bounding rectangle) of all
    radiologist boxes for each finding per image.

    Returns: image_id → {concept_idx → (x1, y1, x2, y2)}
    """
    result = {}
    for iid, findings in bbox_dict.items():
        result[iid] = {}
        for finding, boxes in findings.items():
            if finding not in FINDING_TO_IDX:
                continue
            idx = FINDING_TO_IDX[finding]
            xs1 = [b[0] for b in boxes]
            ys1 = [b[1] for b in boxes]
            xs2 = [b[2] for b in boxes]
            ys2 = [b[3] for b in boxes]
            result[iid][idx] = (min(xs1), min(ys1), max(xs2), max(ys2))
    return result


# ── DICOM conversion ──────────────────────────────────────────────────────────

def dicom_to_png(dicom_path: str, out_path: str, size: int = 1024) -> bool:
    """
    Convert a DICOM file to a grayscale→RGB PNG.
    Applies MONOCHROME1 inversion and global min-max normalisation.
    Returns True on success.
    """
    try:
        import pydicom
        dcm = pydicom.dcmread(dicom_path)
        arr = dcm.pixel_array.astype(np.float32)

        # MONOCHROME1: bright = air (invert so lungs appear dark on white)
        photo = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2").strip()
        if photo == "MONOCHROME1":
            arr = arr.max() - arr

        # Apply VOI LUT if available (windowing)
        try:
            from pydicom.pixel_data_handlers.util import apply_voi_lut
            arr = apply_voi_lut(arr, dcm).astype(np.float32)
        except Exception:
            pass

        # Normalise to [0, 255]
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        arr = arr.clip(0, 255).astype(np.uint8)

        img = Image.fromarray(arr).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        img.save(out_path, "PNG", optimize=True)
        return True
    except Exception as e:
        print(f"  WARN: failed to convert {dicom_path}: {e}")
        return False


def convert_worker(args):
    dicom_path, out_path = args
    if os.path.exists(out_path):
        return True
    return dicom_to_png(dicom_path, out_path)


# ── CSV writers ───────────────────────────────────────────────────────────────

def write_labels_csv(out_path: str, rows: list):
    """rows: list of dicts with keys: image_id, class_label, split, concept_0..N"""
    fieldnames = ["image_id", "class_label", "split"] + \
                 [f"concept_{i}" for i in range(NUM_CONCEPTS)]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_bboxes_csv(out_path: str, rows: list):
    """rows: list of dicts with keys: image_id, concept_idx, x1, y1, x2, y2"""
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id","concept_idx","x1","y1","x2","y2"])
        writer.writeheader()
        writer.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess VinDr-CXR")
    parser.add_argument("--user",     default=os.environ.get("PHYSIONET_USER", ""),
                        help="PhysioNet username (or set PHYSIONET_USER env var)")
    parser.add_argument("--password", default=os.environ.get("PHYSIONET_PASS", ""),
                        help="PhysioNet password (or set PHYSIONET_PASS env var)")
    parser.add_argument("--output",   default=OUTPUT_DIR)
    parser.add_argument("--dicom_dir",default=DICOM_DIR)
    parser.add_argument("--max_train", type=int, default=15000,
                        help="Max training images to download (default: all 15000)")
    parser.add_argument("--max_test",  type=int, default=3000,
                        help="Max test images to download (default: all 3000)")
    parser.add_argument("--workers",   type=int, default=MAX_WORKERS)
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download (DICOMs already in --dicom_dir)")
    parser.add_argument("--image_size", type=int, default=1024,
                        help="Output PNG size (saved at this res; training resizes to 224)")
    args = parser.parse_args()

    if not args.user or not args.password:
        print("ERROR: PhysioNet credentials required.")
        print("  Set PHYSIONET_USER and PHYSIONET_PASS environment variables, or")
        print("  pass --user and --password flags.")
        print()
        print("  To get credentials:")
        print("  1. Register at https://physionet.org/register/")
        print("  2. Sign DUA at https://physionet.org/content/vindr-cxr/1.0.0/")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.dicom_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.dicom_dir, "test"),  exist_ok=True)
    os.makedirs(os.path.join(args.output, "images"),   exist_ok=True)

    random.seed(SEED)

    # ── Step 1: Download annotations ────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Downloading annotation CSVs")
    print("=" * 60)
    ann_paths = download_annotations(args.user, args.password)

    # ── Step 2: Parse annotations ────────────────────────────────────────────
    print("\nStep 2: Parsing annotations")
    print("-" * 40)

    train_votes = parse_global_labels(ann_paths["image_labels_train"])
    test_votes  = parse_global_labels(ann_paths["image_labels_test"])
    bbox_raw    = parse_bbox_annotations(ann_paths["train_bboxes"])

    train_labels = consensus_labels(train_votes)
    test_labels  = consensus_labels(test_votes)
    bbox_consensus = consensus_bboxes(bbox_raw)

    print(f"  Train images: {len(train_labels)}")
    print(f"  Test images:  {len(test_labels)}")
    print(f"  Images with bboxes: {len(bbox_consensus)}")

    # Subset if requested
    train_ids = sorted(train_labels.keys())[:args.max_train]
    test_ids  = sorted(test_labels.keys())[:args.max_test]
    random.shuffle(train_ids)

    # ── Step 3: Create splits ────────────────────────────────────────────────
    print("\nStep 3: Creating splits")
    n_train = int(len(train_ids) * TRAIN_FRAC)
    n_val   = int(len(train_ids) * VAL_FRAC)

    split_map = {}   # image_id → "train" | "val" | "test" | "bbox_eval"

    # Carve out bbox_eval: images in the first portion that have bbox annotations
    bbox_candidates = [iid for iid in train_ids if iid in bbox_consensus][:MAX_BBOX_EVAL]
    bbox_eval_set   = set(bbox_candidates)

    remaining = [iid for iid in train_ids if iid not in bbox_eval_set]
    for iid in remaining[:n_train]:         split_map[iid] = "train"
    for iid in remaining[n_train:n_train+n_val]: split_map[iid] = "val"
    for iid in remaining[n_train+n_val:]:   split_map[iid] = "test"
    for iid in bbox_eval_set:               split_map[iid] = "bbox_eval"
    for iid in test_ids:
        if iid not in split_map:            split_map[iid] = "test"

    all_ids = list(split_map.keys())
    counts = {s: sum(1 for v in split_map.values() if v == s)
              for s in ["train", "val", "test", "bbox_eval"]}
    print(f"  Train: {counts['train']}  Val: {counts['val']}  "
          f"Test: {counts['test']}  BBox eval: {counts['bbox_eval']}")

    # ── Step 4: Download DICOMs ──────────────────────────────────────────────
    if not args.skip_download:
        print("\nStep 4: Downloading DICOMs")
        print(f"  Downloading {len(train_ids)} training + {len(test_ids)} test DICOMs...")
        print("  This will take 1–3 hours depending on connection speed.")

        for split_name, ids in [("train", train_ids), ("test", test_ids)]:
            n_done = 0
            for iid in ids:
                dest = os.path.join(args.dicom_dir, split_name, iid + ".dicom")
                url  = f"{PHYSIONET_BASE}/{split_name}/{iid}.dicom"
                ok   = wget_file(url, dest, args.user, args.password)
                n_done += 1
                if n_done % 500 == 0:
                    print(f"  [{split_name}] {n_done}/{len(ids)} downloaded")
            print(f"  [{split_name}] Done: {n_done}/{len(ids)}")
    else:
        print("\nStep 4: Skipping download (--skip_download)")

    # ── Step 5: Convert DICOMs to PNG ────────────────────────────────────────
    print("\nStep 5: Converting DICOMs to PNG")
    conv_args = []
    for iid, split in split_map.items():
        # Determine which subfolder the DICOM is in
        for sub in ["train", "test"]:
            src = os.path.join(args.dicom_dir, sub, iid + ".dicom")
            if os.path.exists(src):
                dst = os.path.join(args.output, "images", iid + ".png")
                conv_args.append((src, dst))
                break

    print(f"  Converting {len(conv_args)} files with {args.workers} workers...")
    ok_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, result in enumerate(ex.map(convert_worker, conv_args)):
            ok_count += int(result)
            if (i + 1) % 1000 == 0:
                print(f"  Converted {i+1}/{len(conv_args)} ({ok_count} OK)")
    print(f"  Conversion complete: {ok_count}/{len(conv_args)} succeeded")

    # Remove images that failed conversion from split_map
    converted = {Path(p[1]).stem for p in conv_args
                 if os.path.exists(p[1]) and os.path.getsize(p[1]) > 1000}
    split_map = {iid: s for iid, s in split_map.items() if iid in converted}

    # ── Step 6: Write labels.csv ─────────────────────────────────────────────
    print("\nStep 6: Writing labels.csv")
    label_rows = []
    all_labels_dict = {**train_labels, **test_labels}

    for iid, split in split_map.items():
        lbl = all_labels_dict.get(iid, {"concepts": [0]*NUM_CONCEPTS, "class_label": 0})
        row = {
            "image_id":    iid,
            "class_label": lbl["class_label"],
            "split":       split,
        }
        for i, v in enumerate(lbl["concepts"]):
            row[f"concept_{i}"] = v
        label_rows.append(row)

    labels_path = os.path.join(args.output, "labels.csv")
    write_labels_csv(labels_path, label_rows)
    print(f"  Written: {labels_path}  ({len(label_rows)} rows)")

    # ── Step 7: Write bboxes.csv ─────────────────────────────────────────────
    print("\nStep 7: Writing bboxes.csv")
    bbox_rows = []
    for iid in bbox_eval_set:
        if iid not in split_map or iid not in bbox_consensus:
            continue
        for concept_idx, (x1, y1, x2, y2) in bbox_consensus[iid].items():
            bbox_rows.append({
                "image_id": iid,
                "concept_idx": concept_idx,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

    bboxes_path = os.path.join(args.output, "bboxes.csv")
    write_bboxes_csv(bboxes_path, bbox_rows)
    print(f"  Written: {bboxes_path}  ({len(bbox_rows)} bbox rows, "
          f"{len(bbox_eval_set)} images)")

    # ── Step 8: Stats summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VinDr-CXR preprocessing complete!")
    print("=" * 60)
    print(f"  Output directory: {args.output}")
    print(f"  Images:  {len(converted)}")
    print(f"  Labels:  {labels_path}")
    print(f"  Bboxes:  {bboxes_path}")
    print(f"\n  Split breakdown:")
    for s in ["train", "val", "test", "bbox_eval"]:
        n = sum(1 for v in split_map.values() if v == s)
        print(f"    {s:>10}: {n}")

    # Class balance
    n_pos = sum(1 for r in label_rows if r["class_label"] == 1)
    n_neg = len(label_rows) - n_pos
    print(f"\n  Class balance: {n_pos} finding ({n_pos/len(label_rows)*100:.1f}%) "
          f"/ {n_neg} no_finding ({n_neg/len(label_rows)*100:.1f}%)")

    # Concept frequency
    print("\n  Concept frequencies (% of images with finding present):")
    concept_names = [
        "aortic_enlargement", "atelectasis", "calcification", "cardiomegaly",
        "consolidation", "ild", "infiltration", "lung_opacity",
        "nodule_mass", "other_lesion", "pleural_effusion", "pleural_thickening",
        "pneumothorax", "pulmonary_fibrosis",
    ]
    for i, name in enumerate(concept_names):
        freq = sum(1 for r in label_rows if r[f"concept_{i}"] == 1)
        print(f"    {name:<25}: {freq:>5} ({freq/len(label_rows)*100:>5.1f}%)")

    print("\nNext step: train GRAPE on VinDr-CXR:")
    print(f"  python train.py --dataset vindrcxr --data_dir {args.output} "
          f"--config configs/vindrcxr_config.yaml --no_vlm")


if __name__ == "__main__":
    main()
