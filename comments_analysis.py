import os
import random

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import jieba
import jieba.analyse
from wordcloud import WordCloud
from PIL import Image
import tkinter
import networkx as nx
import xlsxwriter


def commentsAnalysis(scriptDirectory, barCode,
                     path_stopwords,
                     path_commentsSentiment_onnx,
                     path_save_commentsAnalysis):
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体 SimHei黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'

    scriptDirectory = scriptDirectory
    barCode = barCode

    # 停用词
    stopwords = []
    path_stopwords = path_stopwords
    with open(path_stopwords, "r", encoding="utf8") as f:
        for w in f:
            stopwords.append(w.strip())

    # 已经分好类的评论
    path_commentsSentiment_onnx = path_commentsSentiment_onnx
    data_comments = pd.read_csv(path_commentsSentiment_onnx)
    df_comments = pd.DataFrame(data_comments)

    # 生成文件夹
    def mkdir(path):
        # 去除首位空格
        path = path.strip()
        # 去除尾部 \ 符号
        path = path.rstrip("\\")
        # 判断路径是否存在
        isExists = os.path.exists(path)
        # 判断结果
        if not isExists:
            os.makedirs(path)
            print(path + ' 创建成功')
            return True
        else:
            # 如果目录存在则不创建，并提示目录已存在
            print(path + ' 目录已存在')
            return False

    # 生成文件夹存储数据分析的结果文件
    path_save_commentsAnalysis = path_save_commentsAnalysis
    mkdir(path_save_commentsAnalysis)

    # Jieba分词函数
    def txt_cut(sentence):
        lis = [word for word in jieba.lcut(sentence) if word not in stopwords]
        return " ".join(lis)

    # 分词并且，添加分词列
    df_comments['cutword'] = df_comments['sentence'].astype('str').apply(txt_cut)
    df_comments.head(5)

    def randomcolor():
        colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
        color = "#" + ''.join([random.choice(colorArr) for i in range(6)])
        return color

    # [randomcolor() for i in range(3)]

    plt.figure(figsize=(5, 5), dpi=180)
    p1 = df_comments['label_emotion'].value_counts()
    plt.pie(p1, labels=p1.index, autopct="%1.3f%%", shadow=True, explode=(0.2, 0),
            colors=[randomcolor() for i in range(2)])  # 带阴影，某一块里中心的距离
    plt.title("情感占比")
    plt.savefig(path_save_commentsAnalysis + '\\情感占比.jpg')
    # plt.show()
    # plt.close()

    # 区分好评和差评
    df_comments_good = df_comments[df_comments["label_emotion"] == 1].reset_index(drop=True)
    df_comments_bad = df_comments[df_comments["label_emotion"] == 0].reset_index(drop=True)
    writer_good = pd.ExcelWriter(path_save_commentsAnalysis + "\\comments_good.xlsx", engine='xlsxwriter')
    df_comments_good.to_excel(writer_good, index=False)
    writer_good._save()

    writer_bad = pd.ExcelWriter(path_save_commentsAnalysis + "\\comments_bad.xlsx", engine='xlsxwriter')
    df_comments_bad.to_excel(writer_bad, index=False)
    writer_bad._save()
    # df_comments_good.to_excel(path_save_commentsAnalysis + "\\comments_good.xlsx", index=False)
    # df_comments_bad.to_excel(path_save_commentsAnalysis + "\\comments_bad.xlsx", index=False)
    # df_comments_good.to_csv(path_save_commentsAn alysis+"\\comments_good.csv", index=False,encoding="utf-8")
    # df_comments_bad.to_csv(path_save_commentsAnalysis+"\\comments_bad.csv", index=False,encoding="utf-8")

    # 文本分析
    # 词频分析
    jieba.analyse.set_stop_words(path_stopwords)
    # 合并一起
    text_good = ''
    text_bad = ''
    for i in range(len(df_comments_good['cutword'])):
        text_good += df_comments_good['cutword'][i] + '\n'
    for i in range(len(df_comments_bad['cutword'])):
        text_bad += df_comments_bad['cutword'][i] + '\n'

    j_r_good = jieba.analyse.extract_tags(text_good, topK=20, withWeight=True)
    j_r_bad = jieba.analyse.extract_tags(text_bad, topK=20, withWeight=True)
    df_wf_good = pd.DataFrame()
    df_wf_bad = pd.DataFrame()
    df_wf_good['word'] = [word[0] for word in j_r_good]
    df_wf_bad['word'] = [word[0] for word in j_r_bad]
    df_wf_good['frequency'] = [word[1] for word in j_r_good]
    df_wf_bad['frequency'] = [word[1] for word in j_r_bad]
    df_wf_good
    df_wf_bad

    # 生成词云
    # Custom colour map based on Netflix palette
    # mask = np.array(Image.open(scriptDirectory + '\\output\\' + barCode + '\\' + barCode + '.png'))
    mask = np.array(Image.open(scriptDirectory + '\\data\\analysis\\money.png'))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [randomcolor() for i in range(20)])
    wordcloud = WordCloud(font_path="C:\\Windows\\Fonts\\simfang.ttf", background_color='white', width=256, height=256,
                          colormap=cmap, max_words=100, mask=mask)

    wordcloud_good = wordcloud.generate(text_good)
    plt.figure(figsize=(10, 6), dpi=512)
    plt.imshow(wordcloud_good, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(path_save_commentsAnalysis + '\\好评词云.jpg')
    # plt.show()
    # plt.close()

    wordcloud_bad = wordcloud.generate(text_bad)
    plt.figure(figsize=(10, 6), dpi=512)
    plt.imshow(wordcloud_bad, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(path_save_commentsAnalysis + '\\差评词云.jpg')
    # plt.show()
    # plt.close()

    # 共现语义网络
    cut_word_list_good = list(map(lambda x: ''.join(x), df_comments_good['cutword'].tolist()))
    cut_word_list_bad = list(map(lambda x: ''.join(x), df_comments_bad['cutword'].tolist()))
    content_str_good = ' '.join(cut_word_list_good).split()
    content_str_bad = ' '.join(cut_word_list_bad).split()
    word_fre_good = pd.Series(tkinter._flatten(content_str_good)).value_counts()  # 统计词频
    word_fre_bad = pd.Series(tkinter._flatten(content_str_bad)).value_counts()  # 统计词频
    word_fre_good[:50]
    word_fre_bad[:50]

    keywords_good = word_fre_good[:50].index
    keywords_bad = word_fre_bad[:50].index
    keywords_good
    keywords_bad

    # 计算矩阵
    matrix_good = np.zeros((len(keywords_good) + 1) * (len(keywords_good) + 1))
    matrix_good = matrix_good.reshape(len(keywords_good) + 1, len(keywords_good) + 1).astype(str)
    matrix_good[0][0] = np.NaN
    matrix_good[1:, 0] = matrix_good[0, 1:] = keywords_good
    matrix_good

    cont_list_good = [cont.split() for cont in cut_word_list_good]
    for i, w1 in enumerate(word_fre_good[:30].index):
        for j, w2 in enumerate(word_fre_good[:30].index):
            count = 0
            # 遍历表中的前五十个词之间在全部文档的关联性
            for cont in cont_list_good:
                if w1 in cont and w2 in cont:
                    # 一个词无论在什么文本跟自己的关联肯定最大，其次是相邻一个位置的词，满足这两个条件则是共现一次
                    # 数值越大代表这个两个词一起出现的概率很大，而且可能出现次数也多
                    # if abs(cont.index(w1) - cont.index(w2)) == 0 or abs(cont.index(w1) - cont.index(w2)) == 1:
                    if abs(cont.index(w1) - cont.index(w2)) == 1:
                        count += 1
            matrix_good[i + 1][j + 1] = count

    matrix_bad = np.zeros((len(keywords_bad) + 1) * (len(keywords_bad) + 1))
    matrix_bad = matrix_bad.reshape(len(keywords_bad) + 1, len(keywords_bad) + 1).astype(str)
    matrix_bad[0][0] = np.NaN
    matrix_bad[1:, 0] = matrix_bad[0, 1:] = keywords_bad
    matrix_bad

    cont_list_bad = [cont.split() for cont in cut_word_list_bad]
    for i, w1 in enumerate(word_fre_bad[:20].index):
        for j, w2 in enumerate(word_fre_bad[:20].index):
            count = 0
            for cont in cont_list_bad:
                if w1 in cont and w2 in cont:
                    # if abs(cont.index(w1) - cont.index(w2)) == 0 or abs(cont.index(w1) - cont.index(w2)) == 1:
                    if abs(cont.index(w1) - cont.index(w2)) == 1:
                        count += 1
            matrix_bad[i + 1][j + 1] = count

    # 存储
    # kwdata = pd.DataFrame(data=matrix)
    # kwdata.to_csv('关键词共现矩阵.csv', index=False, header=None, encoding='utf-8-sig')

    # 查看
    kwdata_good = pd.DataFrame(data=matrix_good[1:, 1:], index=matrix_good[1:, 0], columns=matrix_good[0, 1:])
    kwdata_bad = pd.DataFrame(data=matrix_bad[1:, 1:], index=matrix_bad[1:, 0], columns=matrix_bad[0, 1:])

    # 画图
    plt.figure(figsize=(7, 7), dpi=512)
    graph1_good = nx.from_pandas_adjacency(kwdata_good.iloc[:20, 0:20].astype(int))
    nx.draw(graph1_good, with_labels=True, node_color='blue', font_size=25, edge_color='tomato')
    plt.savefig(path_save_commentsAnalysis + '\\共现网络图-好评.jpg')
    # plt.show()
    # plt.close()

    plt.figure(figsize=(7, 7), dpi=512)
    graph1_bad = nx.from_pandas_adjacency(kwdata_bad.iloc[:20, 0:20].astype(int))
    nx.draw(graph1_bad, with_labels=True, node_color='green', font_size=25, edge_color='tomato')
    plt.savefig(path_save_commentsAnalysis + '\\共现网络图-差评.jpg')
    # plt.show()
    # plt.close()

    # 话题分析
    """好评"""
    # Tf-idf分析，词频逆文档频率
    # 文本转化为词向量
    tf_vectorizer = TfidfVectorizer()
    # tf_vectorizer = TfidfVectorizer(ngram_range=(2,2)) #2元词袋
    X_good = tf_vectorizer.fit_transform(df_comments_good.cutword)
    # print(tf_vectorizer.get_feature_names_out())
    print(X_good.shape)
    # 查看高频词
    data1_good = {'word': tf_vectorizer.get_feature_names_out(),
                  'tfidf': X_good.toarray().sum(axis=0).tolist()}
    df2_good = pd.DataFrame(data1_good).sort_values(by="tfidf", ascending=False, ignore_index=True)
    df2_good.head(20)
    # LDA建模，构建模型，并且拟合
    n_topics_good = 8  # 需要生成的主题数量，这里是八类
    lda_good = LatentDirichletAllocation(n_components=n_topics_good, max_iter=100,
                                         learning_method='batch',
                                         learning_offset=100,
                                         # doc_topic_prior=0.1,
                                         # topic_word_prior=0.01,
                                         random_state=0)
    lda_good.fit(X_good)

    # 查看结果函数
    def print_top_words(model, feature_names, n_top_words):
        tword = []
        tword2 = []
        tword3 = []
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            topic_w = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topic_pro = [str(round(topic[i], 3)) for i in topic.argsort()[:-n_top_words - 1:-1]]  # (round(topic[i],3))
            tword.append(topic_w)
            tword2.append(topic_pro)
            print(" ".join(topic_w))
            print(" ".join(topic_pro))
            print(' ')
            word_pro = dict(zip(topic_w, topic_pro))
            tword3.append(word_pro)
        return tword3

    # 输出每个主题对应词语和概率
    feature_names = tf_vectorizer.get_feature_names_out()
    n_top_words_good = 10
    word_pro_good = print_top_words(lda_good, feature_names, n_top_words_good)
    # 每个评论被分类到哪个主题
    topics_good = lda_good.transform(X_good)
    topics_good = np.argmax(topics_good, axis=1)
    df_comments_good['topic'] = topics_good
    # df.to_excel("data_topic.xlsx",index=False)
    print(topics_good.shape)
    print(df_comments_good.head(5))

    # 通过词云图可视化函数
    def generate_wordcloud(tup):
        color_list = [randomcolor() for i in range(10)]  # 随机生成10个颜色
        wordcloud = WordCloud(background_color='white', font_path='simhei.ttf',  # mask = mask, #形状设置
                              max_words=10, max_font_size=50, random_state=42,
                              colormap=colors.ListedColormap(color_list)  # 颜色
                              ).generate(str(tup))
        return wordcloud

    # 词云显示
    dis_cols = 4  # 一行几个
    dis_rows = 3
    dis_wordnum = 10
    plt.figure(figsize=(5 * dis_cols, 5 * dis_rows), dpi=128)
    kind_good = len(df_comments_good['topic'].unique())

    for i in range(kind_good):
        ax = plt.subplot(dis_rows, dis_cols, i + 1)
        theme_title = list(word_pro_good[i].keys())[0]
        most10 = [(k, float(v)) for k, v in word_pro_good[i].items()][:dis_wordnum]  # 高频词
        ax.imshow(generate_wordcloud(most10), interpolation="bilinear")
        ax.axis('off')
        ax.set_title("第{}类话题:{} 前{}词汇".format(i, theme_title, dis_wordnum), fontsize=30)
    plt.tight_layout()
    plt.savefig(path_save_commentsAnalysis + '\\好评话题词云.jpg')
    # plt.show()
    # plt.close()

    """差评"""
    # Tf-idf分析，词频逆文档频率
    # 文本转化为词向量
    tf_vectorizer = TfidfVectorizer()
    # tf_vectorizer = TfidfVectorizer(ngram_range=(2,2)) #2元词袋
    X_bad = tf_vectorizer.fit_transform(df_comments_bad.cutword)
    # print(tf_vectorizer.get_feature_names_out())
    print(X_bad.shape)
    # 查看高频词
    data1_bad = {'word': tf_vectorizer.get_feature_names_out(),
                 'tfidf': X_bad.toarray().sum(axis=0).tolist()}
    df2_bad = pd.DataFrame(data1_bad).sort_values(by="tfidf", ascending=False, ignore_index=True)
    df2_bad.head(20)
    # LDA建模，构建模型，并且拟合
    n_topics_bad = 8  # 需要生成的主题数量，这里是八类
    lda_bad = LatentDirichletAllocation(n_components=n_topics_bad, max_iter=100,
                                        learning_method='batch',
                                        learning_offset=100,
                                        # doc_topic_prior=0.1,
                                        # topic_word_prior=0.01,
                                        random_state=0)
    lda_bad.fit(X_bad)

    # 查看结果函数
    def print_top_words(model, feature_names, n_top_words):
        tword = []
        tword2 = []
        tword3 = []
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            topic_w = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topic_pro = [str(round(topic[i], 3)) for i in topic.argsort()[:-n_top_words - 1:-1]]  # (round(topic[i],3))
            tword.append(topic_w)
            tword2.append(topic_pro)
            print(" ".join(topic_w))
            print(" ".join(topic_pro))
            print(' ')
            word_pro = dict(zip(topic_w, topic_pro))
            tword3.append(word_pro)
        return tword3

    # 输出每个主题对应词语和概率
    feature_names = tf_vectorizer.get_feature_names_out()
    n_top_words_bad = 10
    word_pro_bad = print_top_words(lda_bad, feature_names, n_top_words_bad)
    # 每个评论被分类到哪个主题
    topics_bad = lda_bad.transform(X_bad)
    topics_bad = np.argmax(topics_bad, axis=1)
    df_comments_bad['topic'] = topics_bad
    # df.to_excel("data_topic.xlsx",index=False)
    print(topics_bad.shape)
    print(df_comments_bad.head(5))

    # 词云显示
    dis_cols = 4  # 一行几个
    dis_rows = 3
    dis_wordnum = 10
    plt.figure(figsize=(5 * dis_cols, 5 * dis_rows), dpi=128)
    kind_bad = len(df_comments_bad['topic'].unique())

    for i in range(kind_bad):
        ax = plt.subplot(dis_rows, dis_cols, i + 1)
        theme_title = list(word_pro_bad[i].keys())[0]
        most10 = [(k, float(v)) for k, v in word_pro_bad[i].items()][:dis_wordnum]  # 高频词
        ax.imshow(generate_wordcloud(most10), interpolation="bilinear")
        ax.axis('off')
        ax.set_title("第{}类话题:{}\n前{}词汇:".format(i, theme_title, dis_wordnum), fontsize=30)
    plt.tight_layout()
    plt.savefig(path_save_commentsAnalysis + '\\差评话题词云.jpg')
    # plt.show()
    # plt.close()


# scriptDirectory = "F:\\pythonProject\\Product_Review_Analysis"
# barCode = "6923450656181"
# commentsAnalysis(scriptDirectory=scriptDirectory,
#                  barCode=barCode,
#                  path_stopwords=scriptDirectory + "\\data\\analysis\\stopwords.txt",
#                  # 已经分好类的评论
#                  path_commentsSentiment_onnx=scriptDirectory + "\\output\\" + barCode + "\\commentsSentiment_onnx.csv",
#                  # 生成文件夹,用于存储数据分析的结果文件
#                  path_save_commentsAnalysis=scriptDirectory + '\\' + "output" + '\\' + barCode + '\\' + "commentsAnalysis")
