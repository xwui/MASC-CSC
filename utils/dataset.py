from torch.utils.data import Dataset
from pathlib import Path

from utils.log_utils import log

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


class CSCDataset(Dataset):
    """
    通用 CSC 数据集加载器，支持从 CSV 文件加载 (src, tgt) 句对。

    参数：
        data_file (str): 数据文件名（相对于 datasets/ 目录的路径），
                         例如 'wang271k_sighan_train.csv' 或 'cscd_ns_train.csv'。
        filepath (str or Path): 可选，如果指定则直接使用该路径加载，忽略 data_file 的默认路径推断。
        limit_size (int): 若大于 0，则截断数据集至该条数（用于快速调试）。
        allow_unequal_length (bool): 是否保留 src 和 tgt 长度不等的样本。
                                     默认为 False（过滤掉）。
                                     对于 CSCD-NS 等包含词级错误的数据集，应设为 True。
    """

    def __init__(self, data_file: str, filepath=None, limit_size=-1,
                 allow_unequal_length: bool = False, **kwargs):
        super(CSCDataset, self).__init__()
        self.data_name = data_file.replace(".csv", "")
        self.allow_unequal_length = allow_unequal_length

        if filepath is not None:
            filepath = filepath
        else:
            filepath = self.get_filepath(data_file)

        self.data, self.error_data = self.load_data_from_csv(filepath)

        if limit_size > 0:
            self.data = self.data[:limit_size]
            log.info("Dataset truncated to %d samples (limit_size=%d).", len(self.data), limit_size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def load_data_from_csv(self, filepath):
        log.info("Load dataset from %s", filepath)
        with open(filepath, mode='r', encoding='utf-8') as f:
            lines = f.readlines()

        data = []
        error_data = []
        for line in lines[1:]:
            items = line.split(",")
            if len(items) < 2:
                continue
            src = items[0].strip()
            tgt = items[1].strip()

            src = ''.join(src.replace(" ", "").replace(u"\u3000", ""))
            tgt = ''.join(tgt.replace(" ", "").replace(u"\u3000", ""))

            if not src or not tgt:
                continue

            if len(src) == len(tgt):
                data.append((src, tgt))
            elif self.allow_unequal_length:
                # CSCD-NS 等含词级错误的数据集，src/tgt 长度可能不等
                # 此类样本作为正常数据保留，但不参与 character-level 对齐评估
                data.append((src, tgt))
            else:
                error_data.append((src, tgt))

        log.info(
            "Load completed. Success num: %d, Skipped (unequal length) num: %d.",
            len(data), len(error_data)
        )

        return data, error_data

    def get_filepath(self, data_name):
        return ROOT / 'datasets' / data_name