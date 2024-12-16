from tools.read_tools import read_groundtruth, read_vectors


def data_check(data_file_path, queries_file_path):

    print('=' * 30)

    # index
    num_data, data_dim, data_vectors = read_vectors(data_file_path)
    print(f"[Dataset 1.2]   index包含 {num_data} 个向量，维度为 {data_dim}")

    # query
    num_queries, query_dim, query_vectors = read_vectors(queries_file_path)
    print(f"[Dataset 1.2]   query包含 {num_queries} 个向量，维度为 {query_dim}")

    print('=' * 30)


