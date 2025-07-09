from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import os
import csv
from tqdm import tqdm

def find_files(file_path, pattern):
    """
    查找文件
    """
    # 使用os.walk遍历目录，获取所有子目录
    for root, dirs, files in os.walk(file_path):
        # 计算当前目录下匹配到的文件数量
        for file in files:
            if file.endswith(pattern):
                yield os.path.join(root, file)
def protein_sequences_generator(file_paths):
    """
    读取数据
    """
    file_paths = list(file_paths)
    print(f"找到{len(file_paths)}个文件")
    for file_path in tqdm(file_paths,desc='正在读取文件:'):
        with open(file_path, 'r') as file:
            for sentence in file:
                yield sentence.strip()
def train_bpe_tokenizer(corpus, bpe_save_path, vocab_size, min_frequency):
    """
    使用BPE算法训练分词器
    """
    all_seq = list(corpus)
    print("总长度:",len(all_seq))
    # 初始化分词器
    tokenizer = Tokenizer(models.BPE())

    # 添加预处理器、解码器和处理器
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(add_prefix_space=True) #是否添加标识符，如开始和结束

    # 训练BPE模型
    trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"],vocab_size=vocab_size, min_frequency=min_frequency)
    print("开始训练BPE模型...")
    tokenizer.train_from_iterator(all_seq, trainer=trainer)
    # 保存分词器
    print("保存分词器...")
    tokenizer.save(bpe_save_path)
    return tokenizer

if __name__ == "__main__":
    #所有蛋白质序列文件路径以及BPE分词后的输出文件路径
    bpe_save_path = "/home/qiansongping/Experimental_Fish_3/4.BPE/bpe_results/bpe_tokenizer.json"
    input_path = '/home/qiansongping/Experimental_Fish_3/4.BPE/data_txt'
    print('正在查找文件')
    file_paths = find_files(input_path, '.txt')
    print('读取蛋白质数据')
    protein_sequences = protein_sequences_generator(file_paths)
    '''
        训练BPE分词器
        k-mer:保留80%的词有8978个，保留75%的词有8417个
    '''
    tokenizer = train_bpe_tokenizer(protein_sequences, bpe_save_path, vocab_size=9000, min_frequency=300) 
    print('训练BPE分词完成')