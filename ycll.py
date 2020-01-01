from lxml import etree
import re
import jieba
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

tag_dict = {
    "0":[1,0,0],
    "1":[0,1,0],
    "2":[0,0,1]
}

def readxml(address,textlist,taglist):
    xml=etree.parse(address) #读取地址为address的文件
    root=xml.getroot() #获取根节点
    tag_list=root.xpath('//@label') #获得全部标签拿出的标签中有些是‘0”’
    for x in tag_list:
        # print(str(x))
        try:
            # print(re.findall(r"\d",str(x))[0])
            # tag = re.findall(r"\d",str(x))[0]
            tag = re.findall(r"\d",x)[0]
            taglist.append(tag)
        except ValueError as ex1:
            print(x)
    # print(tag_list)
    for node in root.xpath('//Sentence[@label]'):
        textlist.append(node.text) #获得有标签的所有句子
    # textlist = root.xpath('//Sentence[@label]').text
    return tag_list

def get_dictionary(all_text):
    # all_ci = ci_juzi_list(all_text)[0]
    # print(all_ci[0])
    # all_word = list(set(all_ci))
    # print(all_word[0])
    # print(len(all_word))
    word = ci_juzi_list(all_text)[0]
    # print(len(word))
    all_word = []
    # print(len(all_word))
    dictionary = {}
    # print(0)
    # print(all_word[0])
    # all_word = list(set(word))
    # for i in word:
    #     if i not in all_word:
    #         all_word.append(i)
    # print(len(all_word))
    # for i in range(all_word.__len__())
    for i in word:
        # if (word.count(i)>5) and (i not in all_word):
        if i not in all_word:
            all_word.append(i)
            # print(word.count(i))
    # print(all_work)
    print(len(all_word))
    for i in range(1,all_word.__len__()+1):
        # if all_word[i-1] not in dictionary:
            # print(i)
            # print(all_word[i-1])
        dictionary[all_word[i-1]] = i
    # print(dictionary.items())
    # print(dictionary["别以为"])
    # print(len(dictionary))
    # for k in dictionary.values():
    #     print(k)#也拿到所有value
    # print(dictionary.items())
    return dictionary

def ci_juzi_list(text):
    with open('./四川大学机器智能实验室停用词库.txt',"r") as f:
        stop_words=list(set(f.read().split("\n"))) #导入停用词
    # print(stop_words)
    all_ci = []  #全部词的集合
    all_juzi = []  #全部句子的集合
    all = [] #全部词和句子的集合
    for i in text:
        # print(i)
        text1 = i.split("\n")
        text2 = [re.sub('\u3000','',part) for part in text1] #去掉某一些字符
        # print(text2)
        text3 = [list(jieba.cut(part,cut_all=False)) for part in text2] #分词
        text4 = sum(text3,[]) #分词后合成一个列表
        # text4 = list(jieba.cut(i,cut_all=False))
        # print(text3)
        # text4 = sum(text3,[])
        # for i in text4:#去停用词
        #     if i in stop_words:
        #         text4.pop(i)
        # print(text4)
        for i in range(text4.__len__())[::-1]: #一个个词拿出来，看是否是停用词，或者是否全是数字
            # if text4[i] in stop_words: # 去除停用词
            #     text4.pop(i)
            if text4[i].isdigit():#如果全都是数字
                text4.pop(i)
        all_ci += text4
        all_juzi.append(text4)
    all.append(all_ci)
    all.append(all_juzi)
    # print(all[0])
    # print(all[1])
    # all_ci=list(set(all_ci))
    return all


def get_vector(text,dictionary):
    all_juzi = ci_juzi_list(text)[1]
    # print(all_juzi[0])
    # all_juzi.append(['别以为', '政治', '与', '无关', '有人', '送', '蒙牛', '的', '产品', '我会', '以为', '对方', '要害'])
    # print(all_juzi[0])
    all_vector = [] #训练集的向量表示
    for list in all_juzi:
        vector = []
        try:
            for i in list:
                vector.append(dictionary[i])
        except KeyError as ex:
            print("1")
        all_vector.append(vector)
    # print(all_vector[0])
    all_vector = pad_sequences(all_vector,maxlen=100,padding='pre')# 在序列前填充,100?，默认补0，maxlen：None或整数，为序列的最大长度
    # print(all_vector)
    return all_vector

def tag_vetor_creat(taglist):
    tag_vetor = []
    for i in taglist:
        tag_vetor.append(tag_dict[i])
    tag_vetor = np.array(tag_vetor)
    return tag_vetor


def main():
    train_text = []

    train_tag = []

    readxml("SMP2019_ECISA_Train.xml",train_text,train_tag)

    # print(train_tag)

    dev_text=[]

    dev_tag = []

    readxml("SMP2019_ECISA_Dev.xml",dev_text,dev_tag)

    # print(train_tag_vetor)

    # print(dev_tag_vetor)

    all_text = train_text + dev_text

    # print(all_text)

    dictionary = get_dictionary(all_text)

    print(dictionary)


    train_x = get_vector(train_text,dictionary)

    train_y = tag_vetor_creat(train_tag)

    test_x = get_vector(dev_text,dictionary)

    test_y = tag_vetor_creat(dev_tag)

    # print(train_x)
    #
    # print(train_y)
    #
    # print(test_x)
    #
    # print(test_y)

    # print(all_word)

    with open("./data.pk","wb") as f:
        pickle.dump(train_x,f)
        pickle.dump(test_x,f)
        pickle.dump(train_y,f)
        pickle.dump(test_y,f)

    # with open("./data.pk","rb") as f:
    #     print(pickle.load(f))
    #     print(pickle.load(f))
    #     print(pickle.load(f))
    #     print(pickle.load(f))
main()
