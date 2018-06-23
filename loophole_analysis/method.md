####一：数据预处理
1. 利用Tokenizer进行分词，符号也算词，将内容全部小写,最后将所有代码都转为sequences
2. 建立Embedding matrix(如何建立词典？×)
3. 二分类，类别one-hot编码，使用损失函数注意使用binary_entrophy
4. 利用word2vec將tokens转换为vector

####二：使用ＢLSTM创建模型