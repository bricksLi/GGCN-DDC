import numpy as np
from munkres import Munkres

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear

from sklearn import metrics


# similar to https://github.com/karenlatong/AGC-master/blob/master/metrics.py，AGE中的评价函数也跟这很相似



def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)  # 8

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    # if numclass1 != numclass2:
    #     for i in l1:
    #         if i in l2:
    #             pass
    #         else:
    #             y_pred[ind] = i
    #             ind += 1
    num_of_iter=0
    while(numclass1 != numclass2):
        if num_of_iter <10:

            num_of_iter+=1
            for i in l1:
                if i in l2:
                    pass
                else:
                    y_pred[ind] = i
                    ind += 1
            l2 = list(set(y_pred))
            numclass2 = len(l2)

        else:

            break
            
    
    cost = np.zeros((numclass1, numclass2), dtype=int)  # (8,8)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average="macro")
    precision_macro = metrics.precision_score(y_true, new_predict, average="macro")
    recall_macro = metrics.recall_score(y_true, new_predict, average="macro")
    f1_micro = metrics.f1_score(y_true, new_predict, average="micro")
    precision_micro = metrics.precision_score(y_true, new_predict, average="micro")
    recall_micro = metrics.recall_score(y_true, new_predict, average="micro")
    return acc, f1_macro,y_pred


# def cluster_acc_sse(y_true, y_pred):  # y_true：真实标签，都是nparray
#     y_true = y_true - np.min(y_true)  # np.min(y_true)==0,

#     l1 = list(set(y_true))
#     numclass1 = len(l1)  # 8

#     l2 = list(set(y_pred))
#     numclass2 = len(l2)

#     ind = 0
#     num_of_iter=0
#     while(numclass1 != numclass2):  # 这里由于vali太少，可能会出现，补全的头两个又覆盖了已有的，导致还是不相等，那就再次添加
#         if num_of_iter <20:
#         # if numclass1 != numclass2:  # 初始时可能会出现数目不等，就做个手动填补，直到最后应该会对应的上
#             num_of_iter+=1
#             for i in l1:
#                 if i in l2:
#                     pass
#                 else:
#                     y_pred[ind] = i
#                     ind += 1
#             l2 = list(set(y_pred))
#             numclass2 = len(l2)
#             # print(f"这是第{num_of_iter}次重匹配标签")
#         else:
#             print("无法匹配合适的标签（匹配循环超过20次）")
#             break
  
#     return y_pred  # 返回填补后的y，即只要标签种类数达到8或者2即可，不需要new_predict


# def eva_sse(y_true, y_pred, X,epoch=0,cate="train"):
#     # 计算轮廓系数作为肘部法则的评价指标
#     print("这是第几个epoch以及是训练集验证集或者测试集：",epoch,cate)
#     if len(set(y_pred.tolist()))<2:  # 刚开始会出现kl散度预测的标签全为一类的情况，大约epoch=10就不会再出现，因为只两类，很容易划分出来
#         print("类别只有一类：直接返回0")
#         return 0
#     elif len(set(y_pred.tolist()))==2:
#         new_predict= cluster_acc_sse(y_true, y_pred)   # 如果大于两类就不要再做这个了;重新凑成3类
       
#         sse = metrics.silhouette_score(X, new_predict, metric='euclidean')   # 使用这个作为计算sse，会由于KL生成的真实标签只有1种，而报错
#         print("生成类别只有两类的sse:",sse)
#         return sse
#     elif len(set(y_pred.tolist()))==3:
#         print("y_pred的内容信息：",epoch,type(y_pred),y_pred.shape,y_pred)
#         sse = metrics.silhouette_score(X, y_pred, metric='euclidean')
        
#         print(f"生成类别有三类的sse {sse:.4f}")
   
#         return sse


def eva(y_true, y_pred,epoch=0,cate="train"):
    
    acc, f1,new_predict = cluster_acc(y_true, y_pred)
    
    nmi = nmi_score(y_true, y_pred, average_method="geometric")
    ari = ari_score(y_true, y_pred)
   
    
    if cate=="last_test" or cate=="pretrain" or epoch % 5 ==0:
        print(f"epoch {cate}_{epoch}:acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}")
    else:
        pass
    return acc, nmi, ari, f1

def eva_t(y_true, y_pred):
    
    acc, f1,new_predict = cluster_acc(y_true, y_pred)
    
    nmi = nmi_score(y_true, y_pred, average_method="geometric")
    ari = ari_score(y_true, y_pred)

    return acc, nmi, ari, f1

