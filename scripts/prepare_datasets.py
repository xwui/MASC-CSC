"""
prepare_datasets.py
====================
统一数据集准备脚本：将 Wang271K/SIGHAN (pickle) 和 CSCD-NS/IME (TSV)
的原始格式统一转换为项目使用的 CSV 格式（src,tgt），放入 datasets/ 目录。

使用方法（在项目根目录运行）：
    python scripts/prepare_datasets.py

数据文件要求（需提前手动上传到服务器）：
    datasets/data/trainall.times2.pkl     <- Wang271K + SIGHAN 合并训练集
    datasets/data/test.sighan13.pkl       <- SIGHAN 2013 测试集
    datasets/data/test.sighan14.pkl       <- SIGHAN 2014 测试集
    datasets/data/test.sighan15.pkl       <- SIGHAN 2015 测试集
    datasets/cscd-ns/train.txt            <- CSCD-NS 训练集 (TSV: label\torigin\tcorrected)
    datasets/cscd-ns/test.txt             <- CSCD-NS 测试集
    datasets/cscd-ime/train.txt           <- CSCD-IME 训练集 (可选, TSV 格式同上)
"""

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def log(msg: str):
    print(f"[prepare_datasets] {msg}", flush=True)


def write_csv(rows: list, output_path: Path, allow_unequal_length: bool = False) -> dict:
    """将 (src, tgt) 对列表写成 CSV，返回统计字典。"""
    total = 0
    written = 0
    skipped_comma = 0
    skipped_length = 0
    has_error_count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode="w", encoding="utf-8") as f:
        f.write("src,tgt\n")
        for src, tgt in rows:
            total += 1
            # 清理空白字符
            src = src.replace(" ", "").replace("\u3000", "").strip()
            tgt = tgt.replace(" ", "").replace("\u3000", "").strip()

            if not src or not tgt:
                continue
            if "," in src or "," in tgt:
                skipped_comma += 1
                continue
            if not allow_unequal_length and len(src) != len(tgt):
                skipped_length += 1
                continue
            if src != tgt:
                has_error_count += 1
            f.write(f"{src},{tgt}\n")
            written += 1

    return {
        "total": total,
        "written": written,
        "skipped_comma": skipped_comma,
        "skipped_length": skipped_length,
        "has_error": has_error_count,
        "error_ratio": f"{has_error_count / written * 100:.1f}%" if written > 0 else "N/A",
    }


def print_stats(name: str, stats: dict):
    log(
        f"  {name}: 共 {stats['total']} 条 → 写入 {stats['written']} 条"
        f"（其中有错误 {stats['has_error']} 条，占比 {stats['error_ratio']}）"
        + (f"，跳过含逗号 {stats['skipped_comma']} 条" if stats["skipped_comma"] else "")
        + (f"，跳过长度不等 {stats['skipped_length']} 条" if stats["skipped_length"] else "")
    )


# ──────────────────────────────────────────────
# Pickle 格式处理 (Wang271K + SIGHAN)
# ──────────────────────────────────────────────

def load_pickle(pkl_path: Path) -> list:
    """加载 pkl 文件，返回 (src, tgt) 元组列表。
    
    兼容两种常见格式：
    1. List[dict]，每个 dict 含 'src'/'tgt' 键  (ReaLiSe 格式)
    2. List[tuple]，每个 tuple 为 (src, tgt)
    """
    if not pkl_path.exists():
        log(f"  ⚠ 找不到文件：{pkl_path}，跳过。")
        return []

    with open(pkl_path, mode="rb") as f:
        data = pickle.load(f)

    rows = []
    for item in data:
        if isinstance(item, dict):
            src = item.get("src", item.get("original", ""))
            tgt = item.get("tgt", item.get("corrected", ""))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            src, tgt = item[0], item[1]
        else:
            continue
        if src and tgt:
            rows.append((str(src), str(tgt)))
    return rows


def process_pickle_datasets():
    """处理所有 pickle 格式的数据集。"""
    data_dir = DATASETS_DIR / "data"
    if not data_dir.exists():
        log(f"⚠ 目录不存在：{data_dir}")
        log("  请先将 .pkl 文件上传到 datasets/data/ 目录，然后重新运行此脚本。")
        return False

    success = True

    # ── Wang271K + SIGHAN 合并训练集 ──
    train_pkl = data_dir / "trainall.times2.pkl"
    rows = load_pickle(train_pkl)
    if rows:
        out = DATASETS_DIR / "wang271k_sighan_train.csv"
        stats = write_csv(rows, out)
        print_stats("✓ wang271k_sighan_train.csv", stats)
    else:
        log("  ✗ 未生成 wang271k_sighan_train.csv（缺少 trainall.times2.pkl）")
        success = False

    # ── SIGHAN 2013/14/15 测试集 ──
    for year, pkl_name, csv_name in [
        ("2013", "test.sighan13.pkl", "sighan_2013_test.csv"),
        ("2014", "test.sighan14.pkl", "sighan_2014_test.csv"),
        ("2015", "test.sighan15.pkl", "sighan_2015_test.csv"),
    ]:
        pkl_path = data_dir / pkl_name
        rows = load_pickle(pkl_path)
        if rows:
            out = DATASETS_DIR / csv_name
            stats = write_csv(rows, out)
            print_stats(f"✓ {csv_name}", stats)
        else:
            log(f"  ✗ 跳过 SIGHAN {year} 测试集（缺少 {pkl_name}）")

    return success


# ──────────────────────────────────────────────
# TSV 格式处理 (CSCD-NS / CSCD-IME)
# ──────────────────────────────────────────────

def load_tsv(tsv_path: Path) -> list:
    """加载 CSCD TSV 文件，格式：label\\torigin\\tcorrected
    
    label=1 表示句子有错误，label=0 表示句子正确（无需纠错）。
    两类样本都保留（正确样本的 src==tgt）。
    """
    if not tsv_path.exists():
        log(f"  ⚠ 找不到文件：{tsv_path}，跳过。")
        return []

    rows = []
    with open(tsv_path, mode="r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                label, origin, corrected = parts
                rows.append((origin, corrected))
            elif len(parts) == 2:
                # 某些版本只有 origin\tcorrected（无 label 列）
                origin, corrected = parts
                rows.append((origin, corrected))
            else:
                log(f"  ⚠ 第 {line_no} 行格式异常，跳过：{line[:60]}")
    return rows


def _find_cscd_file(directory: Path, split: str) -> Path:
    """在目录中查找指定 split 的文件，同时支持 .txt 和 .tsv 扩展名。"""
    for ext in [".txt", ".tsv"]:
        p = directory / f"{split}{ext}"
        if p.exists():
            return p
    return directory / f"{split}.txt"  # 返回默认路径（即使不存在）


def process_cscd_datasets():
    """处理所有 CSCD 格式的数据集。"""
    any_found = False

    # ── CSCD-NS ──
    cscd_ns_dir = DATASETS_DIR / "cscd-ns"
    for split in ["train", "test", "all"]:
        tsv_path = _find_cscd_file(cscd_ns_dir, split)
        rows = load_tsv(tsv_path)
        if rows:
            csv_name = f"cscd_ns_{split}.csv"
            out = DATASETS_DIR / csv_name
            # CSCD-NS 存在少量词级错误（src 和 tgt 长度不相等），设 allow_unequal_length=True
            stats = write_csv(rows, out, allow_unequal_length=True)
            print_stats(f"✓ {csv_name}", stats)
            any_found = True
        else:
            log(f"  ✗ 跳过 CSCD-NS {split} 集（缺少 {tsv_path}）")

    # ── CSCD-IME (可选，大规模伪数据增强) ──
    cscd_ime_dir = DATASETS_DIR / "cscd-ime"
    tsv_path = cscd_ime_dir / "train.txt"
    rows = load_tsv(tsv_path)
    if rows:
        out = DATASETS_DIR / "cscd_ime_train.csv"
        stats = write_csv(rows, out, allow_unequal_length=True)
        print_stats("✓ cscd_ime_train.csv", stats)
        any_found = True
    else:
        log("  ℹ 未找到 CSCD-IME 数据（可选），跳过。")

    return any_found


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def main():
    log("=" * 60)
    log("MASC-CSC 数据集准备脚本")
    log(f"数据根目录：{DATASETS_DIR}")
    log("=" * 60)

    log("\n[1/2] 处理 Pickle 格式数据集（Wang271K + SIGHAN）...")
    pickle_ok = process_pickle_datasets()

    log("\n[2/2] 处理 TSV 格式数据集（CSCD-NS / CSCD-IME）...")
    cscd_ok = process_cscd_datasets()

    log("\n" + "=" * 60)
    if pickle_ok or cscd_ok:
        log("✅ 数据准备完成！生成的 CSV 文件已保存到 datasets/ 目录。")
        log("\n建议的训练命令：")
        log("  # 阶段一：Wang271K 大规模预训练")
        log("  bash scripts/train_stage1_pretrain.sh")
        log("  # 阶段二：CSCD 混合精调")
        log("  bash scripts/train_stage2_finetune.sh")
        log("  # 或一键全流程：")
        log("  bash train_all.sh")
    else:
        log("❌ 未找到任何原始数据文件，请先上传数据后重新运行。")
        log("   参考计划文档中的下载地址说明。")
    log("=" * 60)


if __name__ == "__main__":
    main()
