import os
from gensim.models import Word2Vec
import logging
import csv

class readcsv:
    def __init__(self,file_path,pattern):
        self.file_path = file_path
        self.pattern = pattern

    def find_files(self):
        # 使用os.walk遍历目录，获取所有子目录
        for root, dirs, files in os.walk(self.file_path):
            # 计算当前目录下匹配到的文件数量
            for file in files:
                if file.endswith(self.pattern):
                    yield os.path.join(root, file)
    
    def __iter__(self):
        for file in self.find_files():
            with open(file, 'r', encoding='utf-8') as csv_file:
                 csv_reader = csv.reader(csv_file, delimiter=',')
                 next(csv_reader) #过滤第一行列名
                 for row in csv_reader:
                    seq = row[2].replace("Ġ", "")
                    yield seq.strip().split()
                
corpus_path = '/home/qiansongping/Experimental_Fish_3/4.BPE/protein_data'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = readcsv(corpus_path, '.csv')
'''sg = 0: CBOW（连续词袋模型）
   sg=1 : skip-gram（跳字模型）
   window: 窗口大小，即考虑当前词和其前后多少个词
   min_count: 词频阈值，低于此阈值的词会被忽略
   workers: 并行训练的线程数
'''
model = Word2Vec(sentences, vector_size=100, window=20, min_count=300, workers=32, sg=0)
model.init_sims(True)
model.save("/home/qiansongping/Experimental_Fish_3/4.BPE/bpe_results/word2vec_model_cbow.model")
print('训练完成！')