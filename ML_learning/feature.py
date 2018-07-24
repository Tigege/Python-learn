from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

tag_list = ['青年 吃货 唱歌',
            '少年 游戏 叛逆',
            '少年 吃货 足球']
countVectorizer = CountVectorizer()  # 若要过滤停用词，可在初始化模型时设置
doc_term_matrix = countVectorizer.fit_transform(tag_list)  # 得到的doc_term_matrix是一个csr的稀疏矩阵
# doc_term_matrix[doc_term_matrix>0]=1 #将出现次数大于0的token置1
# doc_term_matrix.todense()#将稀疏矩阵转化为稠密矩阵
vocabulary = countVectorizer.vocabulary_  # 得到词汇表
print(vocabulary)
