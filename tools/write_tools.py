import os
import numpy as np
import torch
from pathlib import Path

def save_data(vectors_index_tensor, dataset=None, type=None, basedir=None):
    if basedir is None:
        basedir = Path.cwd()
        if dataset is not None:
            basedir = basedir/'dataset'/dataset
            basedir.mkdir(parents=True, exist_ok=True)

    if isinstance(vectors_index_tensor, np.ndarray):
        vectors_index_tensor = torch.tensor(vectors_index_tensor, dtype=torch.float32)

    # 获取数据的维度
    nb = vectors_index_tensor.shape[0]  # 索引向量的数量
    d = vectors_index_tensor.shape[1]  # 向量的维度

    path = os.path.join(basedir, type + '_' + str(nb) + '_' + str(d))
    # 保存索引数据
    with open(path, "wb") as f:
        # 保存 nb 和 d
        np.array([nb, d], dtype='uint32').tofile(f)
        # 保存索引向量
        vectors_index_tensor.numpy().astype('float32').tofile(f)

    print(f"[Dataset 1.1]   {type} data saved to {path}")
    return path