# -*- coding: utf-8 -*-
"""
@brief : 将原始数据的文本内容用jieba分词，选定主题，抽取特征并进行向量化，通过朴素贝叶斯分类器进行训练，
         将测试集的主题预测结果保存至本地
@author: Jeque
"""
import time

import jieba
# 导入所需模块
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import cross_val_score

t_start = time.time()  # 开始计时
# 导入原始训练集和测试集，排除训练集的感情值和感情词列。
df_train = pd.read_csv('train.csv', encoding='utf-8')
df_train.drop(columns=['sentiment_value', 'sentiment_word'], inplace=True)
df_test = pd.read_csv('test_public.csv', encoding='utf-8')
df_test['subject'] = ''  # 增加测试集subject列，并设置空值
# df_train.info()   # 查看训练集
# df_test.info()    # 查看测试集
# df_test.head()    # 查看测试集前几行

#
selected_features = ['content', 'subject']
x_train = df_train[selected_features]
x_test = df_test[selected_features]
y_train = df_train['subject']


# x_train['subject'].value_counts()   # 查看训练集各主题数量
# x_test['subject'].value_counts()    # 查看测试集主题数量，当前主题为空值

# jieba进行分词
def train_word_cut(df_train):
    return " ".join(jieba.cut(df_train))

def test_word_cut(df_test):
    return " ".join(jieba.cut(df_test))

def get_custom_stopwords(stop_words_file):
    with open('stopwordsHIT.txt', 'r', encoding='utf-8') as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list


stop_words_file = "stopwordsHIT.txt"
stopwords = get_custom_stopwords(stop_words_file)
x_train["content_cutted"] = x_train.content.apply(train_word_cut)
# x_train.content_cutted.head()    # 查看训练集分词前几行
x_test["content_cutted"] = x_test.content.apply(test_word_cut)
# x_test.content_cutted.head() # 查看测试集分词前几行

# 特征向量化
max_df = 0.8
min_df = 3
vect = CountVectorizer(max_df=max_df,
                       min_df=min_df,
                       token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',    # 去掉数字列
                       stop_words=frozenset(stopwords))

term_matrix_train = pd.DataFrame(vect.fit_transform(x_train.content_cutted).toarray(), columns=vect.get_feature_names())
# term_matrix_train.head()    #查看特征向量化情况
term_matrix_test = pd.DataFrame(vect.fit_transform(x_test.content_cutted).toarray(), columns=vect.get_feature_names())
term_matrix_test.head()

# 使用pipline简化系统搭建流程，将文本抽取与分类器模型串联起来
# 使用朴素贝叶斯进行分类
pipe = make_pipeline(vect, nb)
cross_val_score(pipe, x_train.content_cutted, y_train, cv=5).mean()
pipe.fit(x_train.content_cutted, y_train)
pipe.predict(x_test.content_cutted)
y_pred = pipe.predict(x_test.content_cutted)
pipe_submission = pd.DataFrame({'content_id': df_test['content_id'], 'content': df_test['content'], 'subject': y_pred})
pipe_submission.to_csv('result_subject.csv', index=False)

# 计时结束
t_end = time.time()
print("已将主题训练完成，共耗时：{}min".format((t_end - t_start) / 60))
