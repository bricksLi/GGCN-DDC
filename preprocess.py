import nltk
nltk.data.find(".")
from nltk.corpus import stopwords

from collections import Counter  # 计数
import re  # 正则表达式替换
import joblib  # 保存文件
import numpy as np

dataset = "R52"

# param   ; 参数，这个应该是把文本处理成分词的形式，
stop_words = set(stopwords.words('english'))
least_freq = 5  # 最少的频率
if dataset == "mr" or "SST" in dataset:  # 如果是这两个数据集就不需要去停用词
    stop_words = set()
    least_freq = 0


# func load texts & labels
def load_dataset(dataset):
    with open(f"corpus/{dataset}.texts.txt", "r", encoding="latin1") as f:
        texts = f.read().strip().split("\n")  # 按换行来切分字符串，输出为列表,7674行，read()未给参数则为取所有行
    with open(f"corpus/{dataset}.labels.txt", "r") as f:
        labels = f.read().strip().split("\n")  # 7674
    return texts, labels


def filter_text(text: str):  # 对(一行)一篇文章的转换：处理一些无意义的字符，将缩写改全等操作，使之适合接下来的处理
    text = text.lower()  # 变小写
    # re.sub():对于输入的一个字符串，利用正则表达式（的强大的字符串处理功能），去实现（相对复杂的）字符串替换处理，然后返回被替换后的字符串
    text = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", text)  # re.sub(pattern, repl, string, count=0, flags=0),这里是替换一些特殊字符
    text = text.replace("'ll ", " will ")  # 字符串的替换，把前面的字符全替换为后面的字符，可以看出，这是为了把缩写的单词也弄成完整的
    text = text.replace("'d ", " would ")
    text = text.replace("'m ", " am ")
    text = text.replace("'s ", " is ")
    text = text.replace("'re ", " are ")
    text = text.replace("'ve ", " have ")
    text = text.replace(" can't ", " can not ")
    text = text.replace(" ain't ", " are not ")
    text = text.replace("n't ", " not ")
    text = text.replace(",", " , ")
    text = text.replace("!", " ! ")
    text = text.replace("(", " ( ")
    text = text.replace(")", " ) ")
    text = text.replace("?", " ? ")
    text = re.sub(r"\s{2,}", " ", text)
    return " ".join(text.strip().split())  # join () 方法用于将序列中的元素以指定的字符连接生成一个新的字符串。


if __name__ == '__main__':
    texts, labels = load_dataset(dataset)

    # handle texts，处理文本
    texts_clean = [filter_text(t) for t in texts]  # 清洗数据，7674
    word2count = Counter([w for t in texts_clean for w in t.split()])  # Counter，内置的计数函数，可以统计列表或者集合中出现的不同词的次数：23585
    # print(word2count),输出的是类似字典，键为对应的单词，值为单词出现的次数,这里Counter内有点类似与双循环的缩写
    word_count = [[w, c] for w, c in word2count.items() if c >= least_freq and w not in stop_words]  # 7688个词
    # print(word_count) # ，这里是为了筛选出，出现频率大于等于5次且不是停顿词的词汇,其中每一项形如['concentrate', 30]
    word2index = {w: i for i, (w, c) in enumerate(word_count)}  # 给这7688个词重新编序号
    # print(word2index,type(word2index))，形如{'cascade': 5856, 'amphenol': 5857, 'decrease': 5858},给词编上索引
    words_list = [[w for w in t.split() if w in word2index] for t in texts_clean]  # w是属于Wordindex的键
    # print(words_list,len(words_list))  # 7674,其中每一项是一个列表，列表中是一行的元素中属于7688个词里面的词,且这个列表中每个元素都是str
    texts_remove = [" ".join(ws) for ws in words_list]  # 7674，就是把每一行属于7688词里面的词再重新用空格连接成字符串，再聚成列表
    # print(texts_remove,len(texts_remove)) # 7674个句子，这是把其中的每个子列表组合起来，形成一个新列表，每一项是一个字符串=句子，一行

    # labels 2 targets，处理标签
    label2index = {l: i for i, l in enumerate(set(labels))}  # set(),因为集合不允许重复，可以简单的统计数目，8种标签（i），每种是叫什么名字(l)
    # {'trade': 0, 'acq': 1, 'crude': 2, 'money-fx': 3, 'interest': 4, 'grain': 5, 'ship': 6, 'earn': 7}
    targets = [label2index[l] for l in labels]  # 将7674个标签转为对应的索引，
    # save，将上面处理后的成句单词写成.txt格式
    with open(f"temp/{dataset}.texts.clean.txt", "w") as f:  # 仅仅是过滤后(把缩写扩展，把部分非法词转换)的7674个句子
        f.write("\n".join(texts_clean))

    with open(f"temp/{dataset}.texts.remove.txt", "w") as f:  # 是在过滤的基础上去掉了停顿词后又筛选了属于7688个想要的词里面的7674个句子，
        # 可以比较texts_clean[0]和texts_remove[0]长度发现区别,两者虽都是第一篇文章的单词，但是后者比前者少，因为被过滤了
        f.write("\n".join(texts_remove))

    np.save(f"temp/{dataset}.targets.npy", targets)  # 7674个句子的标签，保存标签，里面是对应的索引数字，而不是具体的标签单词
    joblib.dump(word2index, f"temp/{dataset}.word2index.pkl")  # 把7688个主要词，及其对应的新编号（索引）保存，joblib.dump：保存为序列.pkl形式
