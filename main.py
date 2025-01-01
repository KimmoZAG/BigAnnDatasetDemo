from tools.read_tools import read_fvecs
from tools.write_tools import save_data
from tools.examine_data import data_check
import h5py

'''
读取 index 与 query 数据
numpy.ndarray
(10000, 128)
(100, 128)
'''


# (1,000,000, 128)  (10,000, 128)
datasetname = 'SIFT'
file_path = "/home/kimmo/PycharmProjects/dataset_demo/rawdataset/sift1M/sift/sift_base.fvecs"
index_data = read_fvecs(file_path)
file_path = "/home/kimmo/PycharmProjects/dataset_demo/rawdataset/sift1M/sift/sift_query.fvecs"
query_data = read_fvecs(file_path)


# trevi: index (99900, 4096) query (200, 4096)
# datasetname = 'Trevi'
# file_path = '/home/kimmo/PycharmProjects/dataset_demo/rawdataset/datasets/hdf5/trevi/trevi.hdf5'

# (992272, 420) (200, 420)
# datasetname = 'Msong'
# file_path = '/home/kimmo/PycharmProjects/dataset_demo/rawdataset/datasets/hdf5/msong/msong.hdf5'

# (79106, 512) (200, 420)
# datasetname = 'Sun'
# file_path = '/home/kimmo/PycharmProjects/dataset_demo/rawdataset/datasets/hdf5/sun/sun.hdf5'

# (1192514, 100) (200, 100)
# datasetname = 'Glove'
# file_path = '/home/kimmo/PycharmProjects/dataset_demo/rawdataset/datasets/hdf5/glove/glove.hdf5'

# 打开 HDF5 文件并读取指定数据集
# with h5py.File(file_path, 'r') as hdf:
#     # 读取数据集
#     index_data = hdf['dataset'][:]
#     query_data = hdf['query'][:]


# 确认数据加载成功并检查类型
print("Dataset shape:", index_data.shape)
print("Query shape:", query_data.shape)
print("Dataset type:", type(index_data))
print("Query type:", type(query_data))

q_path = save_data(query_data, datasetname, 'queries')
i_path = save_data(index_data, datasetname, 'data')

data_check(i_path, q_path)