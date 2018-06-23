# coding=utf-8
import os
from keras.preprocessing.text import Tokenizer
from gensim.models import word2vec
from keras.preprocessing.text import text_to_word_sequence
import csv

BASE_DIR = "/home/sancica/Desktop"
TEXT_DATA_DIR = BASE_DIR + "/VulDeePecker/"
MAX_NB_WORDS = 20000
# 先添加所有的代码和标记信息

# 获取文件数据和标签
print("Processing text dataset...")
texts = []
labels = []
for name in os.listdir(TEXT_DATA_DIR):
    fpath = os.path.join(TEXT_DATA_DIR, name)
    if not os.path.isdir(fpath):
        program = []
        with open(fpath)as fr:
            lines = fr.readlines()
            for row in range(len(lines)):
                if "-------" not in lines[row]:
                    program.append(lines[row].strip("\n"))
                else:
                    labels.append(int(program[-1]))
                    program.pop(0)
                    program.pop(-1)
                    texts.append(" ".join(program))
                    program = []
print("Found %s programs." % len(texts))

texts = text_to_word_sequence(" ".join(texts), filters='!"#$%&()*+,./:;<=>?@[]^`|~')
with open("word1.txt", "w")as fw:
    fw.write(" ".join(texts))

texts = list(set(texts))
i = 0
sequences = word2vec.Text8Corpus("./word1.txt")
model = word2vec.Word2Vec(sequences, size=150, min_count=1)
with open("word2vector1.txt", "w")as fw:
    writer = csv.writer(fw, delimiter=" ")
    for word in texts:
        # print(model[word].tolist())
        try:
            print(i)
            i += 1
            writer.writerow([word] + model[word].tolist())
        except Exception:
            print(word)
