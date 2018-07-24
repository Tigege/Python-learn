from sklearn.feature_extraction.text import TfidfVectorizer
#若要过滤停用词，可在初始化模型时设置

tag_list = ['买 这套 系统 本来 是 用来 做 我们 公众',
            '少年 游戏 叛逆',
            '少年 吃货 足球']
tfidfVecorizer = TfidfVectorizer(analyzer=lambda x:x.split(' '))#可自己设置解析方法
tfidf_matrix = tfidfVecorizer.fit_transform(tag_list)
#tfidf_matrix.todense()
term2id_dict = tfidfVecorizer.vocabulary_
print(term2id_dict)
print(tfidf_matrix)