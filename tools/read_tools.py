import numpy as np

def read_fvecs(file_path):
    vectors = []
    with open(file_path, 'rb') as f:
        while True:
            # 读取当前向量的维度（第一个int32整数）
            dim_data = np.fromfile(f, dtype=np.int32, count=1)
            if dim_data.size == 0:
                # 如果读取不到数据，说明文件结束
                break

            dim = dim_data[0]
            # 根据读取的维度，接着读取该行的浮点数数据
            vector = np.fromfile(f, dtype=np.float32, count=dim)

            # 将读取的向量添加到列表
            vectors.append(vector)

    # 将向量列表转换为一个二维数组
    vectors = np.vstack(vectors)
    return vectors

def read_groundtruth(file_path):
    with open(file_path, "rb") as f:
        # 读取文件头信息
        header = np.fromfile(f, dtype='uint32', count=2)  # 读取两个uint32: nq 和 100
        nq, k = header[0], header[1]

        # 读取最近邻索引 I (nq * k 个 uint32)
        I = np.fromfile(f, dtype='uint32', count=nq * k).reshape(nq, k)

        # 读取最近邻距离 D (nq * k 个 float32)
        D = np.fromfile(f, dtype='float32', count=nq * k).reshape(nq, k)

    return nq, k, I, D



def read_vectors(file_path):
    """
    读取二进制文件，返回数据集/查询集的信息和向量。
    """
    with open(file_path, "rb") as f:
        # 读取文件头信息
        header = np.fromfile(f, dtype='uint32', count=2)  # [nb/nq, d]
        num_vectors, dim = header[0], header[1]

        # 读取向量数据
        vectors = np.fromfile(f, dtype='float32').reshape(num_vectors, dim)

    return num_vectors, dim, vectors