import pandas as pd
import numpy as np
import gensim
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from itertools import islice
from gensim.models import Word2Vec
from multiprocessing import Pool

# 2. 数据预处理
def preprocess_sample(sample, word2vec_model):
    sentences_per_sample = sample.split("<end>")
    sentences_per_sample = ['<start> '+ sentence.replace("Ġ", "").strip() +' <end>' for sentence in sentences_per_sample]
    
    sentences_vectors = []
    for sentence in sentences_per_sample:
        words = sentence.split()
        word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
        if word_vectors:
            sentence_vector = np.mean(word_vectors, axis=0) # 句子向量取平均
        else:
            sentence_vector = np.zeros(word_vectors.shape[1]) # 如果句子中没有匹配的词，使用零向量
        sentences_vectors.append(sentence_vector)
    return sentences_vectors

def preprocess_data(data, word2vec_model, max_sentences, file_type, file_path):
    labels = data["Label"].tolist()
    
    with Pool(processes=16) as pool:
        results = list(tqdm(pool.starmap(preprocess_sample, [(sample, word2vec_model) for sample in data["Sequence"]]), total=len(data)))

    sentences = []
    lengths = []
    for result in results:
        length = len(result)
        if length >= max_sentences:
            length = max_sentences
        lengths.append(length)
        
        # 截断或填充句子以满足max_sentences
        if len(result) < max_sentences:
            padding = [np.zeros(word2vec_model.vector_size)] * (max_sentences - len(result))
            result.extend(padding)
        else:
            result = result[:max_sentences]
        sentences.append(result)

    # 将生成的张量保存为文件
    tensor_filename = file_path +'/'+ file_type +"_sentences_tensor.pth"
    label_filename = file_path +'/'+ file_type+"_labels_tensor.pth"
    length_filename = file_path +'/'+ file_type+"_lengths_tensor.pth"
    
    torch.save(torch.tensor(sentences), tensor_filename)
    torch.save(torch.tensor(labels), label_filename)
    torch.save(torch.tensor(lengths), length_filename)

    print("ALL Done! Tensors saved to files.")

    return torch.tensor(sentences), torch.tensor(labels)

if __name__ == "__main__":
    train_data = pd.read_csv("/home/qiansongping/Experimental_Fish_3/5.Data_padding/Data_merge1/Seqence_merge_Label_train.csv")  # 假设您的训练数据文件名为"train.csv"
    val_data = pd.read_csv("/home/qiansongping/Experimental_Fish_3/5.Data_padding/Data_merge1/Seqence_merge_Label_val.csv")  # 假设您的验证数据文件名为"validation.csv"
    test_data = pd.read_csv("/home/qiansongping/Experimental_Fish_3/5.Data_padding/Data_merge1/Seqence_merge_Label_test.csv")  # 假设您的测试数据文件名为"test.csv"
    # 3. 加载预训练Word2Vec模型
    word2vec_model = gensim.models.KeyedVectors.load("/home/qiansongping/Experimental_Fish_3/4.BPE/bpe_results/word2vec_model_cbow.model")  # 假设您的Word2Vec模型文件名为"word2vec_model.bin"
    # 4. 预处理训练集、验证集和测试集
    file_path = '/home/qiansongping/Experimental_Fish_3/6.train1/Vectors_21000'
    print("Preprocessing train data...")
    train_X, train_y = preprocess_data(train_data, word2vec_model, 21000, 'train', file_path)
    print("Preprocessing test data...")
    test_X, test_y = preprocess_data(test_data, word2vec_model, 21000, 'test', file_path)
    print("Preprocessing val data...")
    val_X, val_y = preprocess_data(val_data, word2vec_model, 21000, 'val', file_path)
