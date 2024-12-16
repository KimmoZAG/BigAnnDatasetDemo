from tools.read_tools import read_fvecs
from tools.write_tools import save_data
from tools.examine_data import data_check
'''
读取 index 与 query 数据
numpy.ndarray
(10000, 128)
(100, 128)
'''

datasetname = 'siftsmall'
file_path = "/home/kimmo/PycharmProjects/dataset_demo/rawdataset/sift10K/siftsmall_base.fvecs"
index_data = read_fvecs(file_path)
file_path = "/home/kimmo/PycharmProjects/dataset_demo/rawdataset/sift10K/siftsmall_query.fvecs"
query_data = read_fvecs(file_path)

q_path = save_data(query_data, datasetname, 'queries')
i_path = save_data(index_data, datasetname, 'data')

data_check(i_path, q_path)