"""
data_process.py（扩展版）
===========================
支持将 pickle(.pkl) 和 TSV 格式的原始数据转换为项目 CSV 格式。
推荐直接使用 prepare_datasets.py 代替此脚本完成完整的数据准备，
本脚本保留以向后兼容原有的简单 pickle 转换功能。

使用方法：
    python scripts/data_process.py               # 处理全部已知数据集
    python scripts/data_process.py --sighan-only # 只处理 SIGHAN 测试集
"""

import argparse
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "data"
OUT_DIR = ROOT / "datasets"


def process_pkl(input_name: str, output_name: str):
    """将 pkl 文件（ReaLiSe 格式）转换为 CSV 文件。"""
    pkl_path = DATA_DIR / input_name
    if not pkl_path.exists():
        print(f"[SKIP] {pkl_path} 不存在，跳过。")
        return

    with open(pkl_path, mode="rb") as f:
        data_list = pickle.load(f)

    out_path = OUT_DIR / output_name
    written = 0
    skipped = 0
    with open(out_path, mode="w", encoding="utf-8") as f:
        f.write("src,tgt\n")
        for item in data_list:
            if isinstance(item, dict):
                src = item.get("src", item.get("original", ""))
                tgt = item.get("tgt", item.get("corrected", ""))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                src, tgt = str(item[0]), str(item[1])
            else:
                continue

            src = src.replace(" ", "").replace("\u3000", "").strip()
            tgt = tgt.replace(" ", "").replace("\u3000", "").strip()

            if not src or not tgt:
                continue
            if len(src) != len(tgt):
                skipped += 1
                continue
            if "," in src or "," in tgt:
                skipped += 1
                continue
            f.write(f"{src},{tgt}\n")
            written += 1

    print(f"[OK] {input_name} → {output_name}  (写入 {written} 条，跳过 {skipped} 条)")


def process_tsv(input_path: Path, output_name: str, allow_unequal_length: bool = True):
    """将 CSCD TSV 格式（label\\torigin\\tcorrected）转换为 CSV 文件。"""
    if not input_path.exists():
        print(f"[SKIP] {input_path} 不存在，跳过。")
        return

    rows = []
    with open(input_path, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                _, origin, corrected = parts
            elif len(parts) == 2:
                origin, corrected = parts
            else:
                continue
            rows.append((origin, corrected))

    out_path = OUT_DIR / output_name
    written = 0
    skipped = 0
    with open(out_path, mode="w", encoding="utf-8") as f:
        f.write("src,tgt\n")
        for src, tgt in rows:
            src = src.replace(" ", "").replace("\u3000", "").strip()
            tgt = tgt.replace(" ", "").replace("\u3000", "").strip()
            if not src or not tgt:
                continue
            if "," in src or "," in tgt:
                skipped += 1
                continue
            if not allow_unequal_length and len(src) != len(tgt):
                skipped += 1
                continue
            f.write(f"{src},{tgt}\n")
            written += 1

    print(f"[OK] {input_path.name} → {output_name}  (写入 {written} 条，跳过 {skipped} 条)")


def _find_cscd_file(directory: Path, split: str) -> Path:
    """同时支持 .txt 和 .tsv 扩展名。"""
    for ext in [".txt", ".tsv"]:
        p = directory / f"{split}{ext}"
        if p.exists():
            return p
    return directory / f"{split}.txt"


def main():
    parser = argparse.ArgumentParser(description="MASC-CSC 数据格式转换脚本")
    parser.add_argument("--sighan-only", action="store_true",
                        help="只处理 SIGHAN 2013/14/15 测试集 (pkl → csv)")
    args = parser.parse_args()

    print("=" * 55)
    print("数据格式转换脚本")
    print("=" * 55)

    # SIGHAN 测试集（所有情况下都处理）
    for pkl_name, csv_name in [
        ("test.sighan13.pkl", "sighan_2013_test.csv"),
        ("test.sighan14.pkl", "sighan_2014_test.csv"),
        ("test.sighan15.pkl", "sighan_2015_test.csv"),
    ]:
        process_pkl(pkl_name, csv_name)

    if not args.sighan_only:
        # Wang271K + SIGHAN 合并训练集
        process_pkl("trainall.times2.pkl", "wang271k_sighan_train.csv")

        # CSCD-NS
        cscd_ns_dir = ROOT / "datasets" / "cscd-ns"
        process_tsv(_find_cscd_file(cscd_ns_dir, "train"), "cscd_ns_train.csv")
        process_tsv(_find_cscd_file(cscd_ns_dir, "test"),  "cscd_ns_test.csv")
        # 如果存在 all.txt/tsv，也一并转换
        all_path = _find_cscd_file(cscd_ns_dir, "all")
        if all_path.exists():
            process_tsv(all_path, "cscd_ns_all.csv")

        # CSCD-IME (可选)
        cscd_ime_dir = ROOT / "datasets" / "cscd-ime"
        process_tsv(_find_cscd_file(cscd_ime_dir, "train"), "cscd_ime_train.csv")

    print("=" * 55)
    print("转换完成，文件已保存到 datasets/ 目录。")


if __name__ == "__main__":
    main()
