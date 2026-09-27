import os
import shutil
import argparse
from pathlib import Path
import yaml

def ensure_style_A(yaml_path: Path):
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    base = Path(cfg.get("path", yaml_path.parent))

    has_A = any((base / "images" / s).exists() for s in ["train", "val", "test"])
    has_B = any((base / s / "images").exists() for s in ["train", "val", "test"])

    if has_A:
        print("Dataset already in A style, no conversion needed.")
        return
    if has_B:
        print("Dataset detected B style, converting to A ...")
        convert_b_to_a(str(base))
        return

    raise RuntimeError(f"Could not detect dataset style under base={base}")


def convert_a_to_b(root):
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)
        if not os.path.exists(img_dir):
            continue

        target_split = os.path.join(root, split)
        os.makedirs(os.path.join(target_split, "images"), exist_ok=True)
        os.makedirs(os.path.join(target_split, "labels"), exist_ok=True)

        for f in os.listdir(img_dir):
            shutil.move(os.path.join(img_dir, f), os.path.join(target_split, "images", f))
        for f in os.listdir(lbl_dir):
            shutil.move(os.path.join(lbl_dir, f), os.path.join(target_split, "labels", f))

    shutil.rmtree(os.path.join(root, "images"), ignore_errors=True)
    shutil.rmtree(os.path.join(root, "labels"), ignore_errors=True)


def convert_b_to_a(root):
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(root, split)
        if not os.path.exists(split_dir):
            continue

        img_dir = os.path.join(split_dir, "images")
        lbl_dir = os.path.join(split_dir, "labels")

        target_img_dir = os.path.join(root, "images", split)
        target_lbl_dir = os.path.join(root, "labels", split)
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_lbl_dir, exist_ok=True)

        for f in os.listdir(img_dir):
            shutil.move(os.path.join(img_dir, f), os.path.join(target_img_dir, f))
        for f in os.listdir(lbl_dir):
            shutil.move(os.path.join(lbl_dir, f), os.path.join(target_lbl_dir, f))

        shutil.rmtree(split_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO dataset format between A and B")
    parser.add_argument("root")
    parser.add_argument("--to", choices=["a", "b"], required=True)
    args = parser.parse_args()

    if args.to == "b":
        convert_a_to_b(args.root)
        print("a to b")
    else:
        convert_b_to_a(args.root)
        print("b to a")
