import joblib
from tqdm import tqdm
import scipy.sparse as sp  # 用于构建稀疏矩阵
from collections import Counter
import numpy as np

# 数据集
# dataset = "R52"
dataset = "R8"

# 参数
window_size = 1
embedding_dim = 300
max_text_len = 800


# normalize
def normalize_adj(adj):  # 利用拉普拉斯标准化D-1/2*A*D-1/2
    row_sum = np.array(adj.sum(1))  # 逐行求和，1，分别为13,12，...(81, 1) 得到D
    with np.errstate(divide='ignore'):  # 忽视有可能的浮点错误
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()  # (81,),这里的标准化1处理,numpy.ndarray
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 如果有无穷大的就另为0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)  # D-1/2，(81, 81)
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)  # 标准化2，这里的.dot类似一般矩阵点乘，.transpose相当于转置
    return adj_normalized


def pad_seq(seq, pad_len):  # 少于顶点max（291）的需要用0替代；少于边数max957的也要用0替代，权重的表示同边
    if len(seq) > pad_len: return seq[:pad_len]  # seq还是一个list，它的元素就是每一行的顶点，其实不会有大雨的
    return seq + [0] * (pad_len - len(seq))


def preprocess_graph(adj, layer, norm='sym', renorm=True):  # AGE中改进的拉普拉斯（代替上面的normalize）：H，因为GNN3层，准备了三个
    adj = sp.coo_matrix(adj)  # (3327,3327)
    ident = sp.eye(adj.shape[0])  # 对角阵，对角元素为1，其余为0，(3327,3327)
    if renorm:  # A+I，允许自连接
        adj_ = adj + ident
    else:  # 不允许自连接：A
        adj_ = adj

    rowsum = np.array(adj_.sum(1))  # 按行求和，(3327,1),D~矩阵：在A~基础上求得

    if norm == 'sym':  # 标准拉普拉斯归一化，Lsym=D-1/2*L*D-1/2=I-D-1/2*A*D-1/2
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())  # D-1/2，(3327,3327)
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized  # L=I-D-1/2*A*D-1/2，(3327,3327)
    elif norm == 'left':  # D-1*L
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized

    reg = [2 / 3] * (layer)  # 这里做了拉普拉斯的修改，乘2/3而非1

    adjs = []
    for i in range(len(reg)):
        adjs.append(ident - (reg[i] * laplacian))  # I-k*L
    return adjs


if __name__ == '__main__':  # 根据预训练后的数据集生成可供构造图的数据
    # load data，加载数据
    word2index = joblib.load(f"temp/{dataset}.word2index.pkl")  # 变为了原来的字典类型，形如：'systems': 2, 'completes': 3,
    with open(f"temp/{dataset}.texts.remove.txt", "r") as f:
        texts = f.read().strip().split("\n")  # 7674行

    # bulid graph，生成图数据，按照共现关系，一篇文档生成一个图
    inputs = []
    graphs = []
    for text in tqdm(texts):  # 使得本循环中的代码进度可视化,且由于texts是可序列化数据，因此每次循环处理一行数据，也即列表的一项
        words = [word2index[w] for w in text.split()]  # 对应一行数据中各个单词在"7688词"索引表中的索引
        words = words[:max_text_len]  # 限制最大长度，即一行最大单词长度不超过800，[0,800),如果不够就相当于不变，这里未凑够800
        nodes = list(set(words))  # 去重，因为接下来要把每个单词作为节点编序，重复单词显然没必要，得到的是索引1--也即数字
        # print(nodes,len(nodes))
        node2index = {e: i for i, e in enumerate(nodes)}  # 这里是构建字典，把单词对应的索引1当做键，把索引1在新生成的列表中的索引2当成值
        # print(node2index) 为何重新又排序，因为这里只是第一篇文章还是按序，之后的可能就会出现不按序，为了构建图的邻接矩阵，要重新排序
        edges = []
        for i in range(len(words)):  # 对这一行单词进行遍历，这里就是相当于对实际出现的边(words共现),都采用nodes（去重后的新编号），正好可以生成adj
            center = node2index[words[i]]  # center作为滑动窗口中心，从头开始，
            for j in range(i - window_size, i + window_size + 1):  # w_s为1
                if i != j and 0 <= j < len(words):  # 前一个条件是为了防止顶点自己与自己相连，后一个是为了防止超过这一行的范围，连错
                    neighbor = node2index[words[j]]  # 邻居索引，因为是在words中查看实际连边的,但是实际邻接矩阵中是看nodes中的新排索引
                    edges.append((center, neighbor))  # 边用中心点和邻居点构成的顶点对表示,注意：因为每个都会连两遍，是无向图
        edge_count = Counter(edges).items()  # 加了items()方法，其中元素为（键，值），会有重复的边，这里把重复次数作为邻接矩阵中那条边的权重，相当于去重
        # 这里是某条边和出现的次数，形如dict_items([((13, 14), 1), ((14, 13), 1), ((14, 15), 1), ((15, 14), 1)])

        # 构建稀疏矩阵==邻接矩阵，值为对应权重
        row = [x for (x, y), c in edge_count]  # 得到的边对的中心点,221,set(row)长为81
        col = [y for (x, y), c in edge_count]  # 得到的是边对的另一顶点.221
        weight = [c for (x, y), c in edge_count]  # 边对出现的次数作为权重
        adj = sp.csr_matrix((weight, (row, col)), shape=(len(nodes), len(nodes)))  # 因为是无向图，所以矩阵尺寸为顶点数*顶点数81*81
        # print(adj,type(adj))  scipy.sparse.csr.csr_matrix，这部分与age中相同格式，只不过加了权重
        adj_normalized = normalize_adj(adj)  # 标准化邻接矩阵，拉普拉斯标准化，且变为了np.ndarray,(81,81)，对应的位置变为指定的权重
        weight_normalized = [adj_normalized[x][y] for (x, y), c in edge_count]  # 取的是标准化后的A中对应边位置的值，也即标准权值，而非c

        inputs.append(nodes)  # 输入的是nodes，是索引1,不是标准化后的
        graphs.append([row, col, weight_normalized])  # 这里graphs就是存储了入点出点和相应的权重：将A标准化后对应的边的值

    # print(type(inputs),len(inputs),type(graphs),len(graphs))  都是列表，都是7674
    len_inputs = [len(e) for e in inputs]  # 每一行对应的顶点数
    # for x, y, c in graphs:
    #   print(x)  # 都是列表，x,是中心节点的顶点集，y是指出的顶点集，c是边权重集，每一个循环对应一行
    #   print(y)
    #   print(c)
    len_graphs = [len(x) for x, y, c in graphs]  # 图中每一行的，也即一行中的去重边的中心顶点集数目(也即边的数目)

    # padding input  ，填补输入的空白，如果不足则补0
    pad_len_inputs = max(len_inputs)  # 返回最大项的值，输入可迭代对象，最大为291个顶点
    pad_len_graphs = max(len_graphs)  # 最多边的文档有957条边
    inputs_pad = [pad_seq(e, pad_len_inputs) for e in tqdm(inputs)]  # e是inputs中一个元素=一行中的顶点，tqdm就理解为加了一个可视化。
    # 填补后的输入，都补成了长度与max一样的列表，去掉了每一行的重复，也去掉了其中不合理的顶点，
    # 下面这个操作：e分别是每一行的[出点，入点和权重列表]
    graphs_pad = [[pad_seq(ee, pad_len_graphs) for ee in e] for e in tqdm(graphs)]

    inputs_pad = np.array(inputs_pad)  # (7674, 291)
    weights_pad = np.array([c for x, y, c in graphs_pad])  # (7674, 957)
    graphs_pad = np.array([[x, y] for x, y, c in graphs_pad])  # (7674, 2, 957)  ，图的数目957与边对有关，所以数目与输入的顶点291不同
    # 也就是说graphs_pad可能有（0,0）边对，这是补的
    # word2vec
    # ①Glove词典
    all_vectors = np.load(f"source/glove.6B.{embedding_dim}d.npy")  # 下载“词典”，数字向量化，类型：numpy.ndarray,(400000, 300)
    all_words = joblib.load(f"source/glove.6B.words.pkl")  # 导入词，单词40000个，类型为列表
    all_word2index = {w: i for i, w in enumerate(all_words)}  # 把单词标上索引，是字典

    # ②导入数据集
    index2word = {i: w for w, i in word2index.items()}  # 数据集字典（7688个词），items() 函数以列表返回可遍历的(键, 值) 元组数组。
    word_set = [index2word[i] for i in range(len(index2word))]  # 列表，每一项为上面的单词，按顺序,7688
    oov = np.random.normal(-0.1, 0.1, embedding_dim)  # (300,)，
    word2vec = [all_vectors[all_word2index[w]] if w in all_word2index else oov for w in word_set]
    # 7689,w是单词。如果w在glove字典里，那么就用glove对应的向量，否则就用oov随机初始化
    word2vec.append(np.zeros(embedding_dim))  # 是一个列表，里面是一个个矩阵元素，维度为(300,)，多加了一行全0

    # save
    joblib.dump(len_inputs, f"temp/{dataset}.len.inputs.pkl")  # 每一行对应的单词的数目
    joblib.dump(len_graphs, f"temp/{dataset}.len.graphs.pkl")  # 输入的图的每一行的边的数目
    np.save(f"temp/{dataset}.inputs.npy", inputs_pad)  # 填补维度后的每篇文章的向量表示，(7674, 291)，
    np.save(f"temp/{dataset}.graphs.npy", graphs_pad)  # 填补维度后的图中边的表示，(7674, 2, 957)
    np.save(f"temp/{dataset}.weights.npy", weights_pad)  # 填补维度后的每篇文章的权重的表示，(7674, 957)
    np.save(f"temp/{dataset}.word2vec.npy", word2vec)  # 7689（7688+全0）每个单词的的向量表示<每个单词的表示也是300维
