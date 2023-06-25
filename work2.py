import random
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np


"""
采用修饰器技术对作业1随机数据结构生成函数进行修饰，实现所有生成随机数的四种机器学习方法（SVM,RF,CNN,RNN）操作，
四种精度指标（ACC,MCC,F1,RECALL）操作，以及相应的调用示例。
主要考察点是带参数修饰器的使用，具体要求如下：
1. 修饰器类型不限，可以是函数修饰器或类修饰器；
2. 实现两个修饰器，通过修饰器参数（*args）实现机器学习方法和验证指标操作的任意组合；
"""


# 机器学习方法修饰器
def ml_decorator(*args):
    def outer_wrapper(func):
        def inner_wrapper(**kwargs):
            global pred_y
            # 获取数据和特征数
            data = list(kwargs.get('data'))
            feature = kwargs.get('feature')

            for model in args:
                print(f'正在执行{model}操作...')
                if model == 'SVM' or model == 'RF':
                    # 将数据变成每行一个sample的形式，方便训练和验证
                    X = flatten(data)
                    X = np.array(X).reshape(len(X) // feature, feature)
                    print(f'每一个sample如下：{X}')
                    X = onehot(X)
                    # print(f'onehot之后的sample={X}')

                    slicing = int(len(X) * 0.7)  # 七成用作训练集

                    clf = func(data=X[0: slicing], feature=feature)
                    pred_y = clf.predict(X[slicing:])
                    print(f'{model}方法预测的验证集标签：{pred_y}')
                elif model == 'CNN' or model == 'RNN':
                    func(**kwargs)
                    samples = len(flatten(data)) // feature
                    slicing = int(samples * 0.7)
                    return [random.randint(0, 1) for _ in range(samples - slicing)]
                print(f'{model}执行完毕...\n')
            return pred_y
        return inner_wrapper
    return outer_wrapper


# 精度指标修饰器
def metrics_decorator(*args):
    def outer_wrapper(func):
        def inner_wrapper(**kwargs):
            for metric in args:
                if metric == 'ACC':
                    print('执行了ACC操作...')
                elif metric == 'MCC':
                    print('执行了MCC操作...')
                elif metric == 'F1':
                    print('执行了F1操作...')
                elif metric == 'RECALL':
                    print('执行了RECALL操作...')
            result = func(**kwargs)
            return result
        return inner_wrapper
    return outer_wrapper


# 随机数据结构生成函数
def dataSampling(**kwargs):
    new_shape, type1 = kwargs.get('shape'), kwargs.get('type')
    print('我想要一个{}维度的随机数据，其中每个元素的类型是{}'.format(new_shape, type1))

    # 计算列表长度，即new_shape的每一个元素乘起来
    data_len = 1
    for _ in new_shape:
        data_len *= _

    # 根据type生成数据
    res = []
    for _ in range(data_len):
        tmp = []
        for data_type in type1:
            if data_type == int:
                tmp.append(random.randint(1, 100))
            elif data_type == float:
                tmp.append(round(random.uniform(0, 100), 3))
            elif data_type == str:
                tmp.append(''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(5, 10))))
            else:
                print(f'unsupported data type: {data_type}')
        res.append(tmp)

    # print('当前获取的随机列表为{}, 我想要的维度是{}.'.format(res, new_shape))
    return reshape(res, new_shape), len(type1)


def reshape(res, new_shape):
    new_len = len(res)

    # 每次拿最后n个元素组成一个列表
    for n in new_shape[::-1]:
        # print("----------")
        lst = []
        for _ in range(new_len // n):   # 拿几次
            tmp = res[-n:]      # 切片，取最后n个元素
            # print("tmp:", tmp)
            lst.insert(0, tmp)  # 头插
            res = res[:-n]      # 剩下的切片重新赋值

        # print("lst:", lst)
        res = lst
        new_len //= n

    return res


def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


def onehot(data):
    encoder = OneHotEncoder()
    data = encoder.fit_transform(data).toarray()
    return data


# 生成样本真实标签的方法
def get_labels(data, feature):
    X = flatten(data)
    X = np.array(X).reshape(len(X) // feature, feature)
    # print("get_labels的data=", X)

    # 找到首个不是str类型元素的索引
    ind = -1
    for i in range(feature):
        if not isinstance(X[i], str):
            ind = i
            break

    # 下一行解释：如果都是str类型元素，随机生成标签
    Y = [random.randint(0, 1) for _ in range(len(X))]
    sum = 0
    if ind != -1:
        for i in range(len(X)):
            sum += int(X[i][ind])
        ave = sum / len(X)
        for i in range(len(X)):
            Y[i] = 1 if int(X[i][ind]) > ave else 0

    return Y[int(len(Y) * 0.7):]


@ml_decorator('SVM')
def svm_method(**kwargs):
    X = kwargs.get('data')
    # print(X)

    # 假设标签为二分类问题，0和1代表两个类别
    y = [random.randint(0, 1) for _ in range(len(X))]
    # print("y = ", y)

    # 创建SVM分类器对象
    clf = svm.SVC()
    clf.fit(X, y)
    return clf


@ml_decorator('RF')
def rf_method(**kwargs):
    X = kwargs.get('data')

    # 假设标签为二分类问题，0和1代表两个类别
    y = [random.randint(0, 1) for _ in range(len(X))]

    # 创建随机森林分类器对象
    clf = RandomForestClassifier()
    clf.fit(X, y)  # 使用训练数据进行模型训练
    return clf


@ml_decorator('CNN')
def cnn_method(**kwargs):
    print("执行了CNN...")


@ml_decorator('RNN')
def rnn_method(**kwargs):
    print("执行了RNN...")


@metrics_decorator('ACC')
def acc_metric(**kwargs):
    y_true = kwargs.get('y_true')
    y_pred = kwargs.get('y_pred')
    accuracy = (y_true == y_pred).mean()
    return accuracy


@metrics_decorator('MCC')
def mcc_metric(**kwargs):
    y_true = kwargs.get('y_true')
    y_pred = kwargs.get('y_pred')
    # y_pred = list(y_pred)
    # print(f'MCC y_true={y_true}')
    # print(f'MCC y_pred={y_pred}')

    tp = sum(1 for x, y in zip(y_true, y_pred) if x == y == 1)
    tn = sum(1 for x, y in zip(y_true, y_pred) if x == y == 0)
    fp = sum(1 for x, y in zip(y_true, y_pred) if x == 0 and y == 1)
    fn = sum(1 for x, y in zip(y_true, y_pred) if x == 1 and y == 0)

    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if denominator == 0:
        return 0
    else:
        return numerator / denominator


@metrics_decorator('F1')
def f1_metric(**kwargs):
    y_true = kwargs.get('y_true')
    y_pred = kwargs.get('y_pred')

    tp = sum(1 for x, y in zip(y_true, y_pred) if x == y == 1)
    fp = sum(1 for x, y in zip(y_true, y_pred) if x == 0 and y == 1)
    fn = sum(1 for x, y in zip(y_true, y_pred) if x == 1 and y == 0)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


@metrics_decorator('RECALL')
def recall_metric(**kwargs):
    y_true = kwargs.get('y_true')
    y_pred = kwargs.get('y_pred')

    tp = sum(1 for x, y in zip(y_true, y_pred) if x == y == 1)
    fn = sum(1 for x, y in zip(y_true, y_pred) if x == 1 and y == 0)

    recall = tp / (tp + fn)
    return recall


# 调用示例
def main():
    data, feature = dataSampling(shape=(4, 2, 2), type=(int, str, float))
    print(f"随机生成的数据如下：\n{data}\n")

    # 调用机器学习方法
    # y_pred = svm_method(data=data, feature=feature)
    y_pred = rf_method(data=data, feature=feature)
    # y_pred = rnn_method(data=data, feature=feature)
    # y_pred = cnn_method(data=data, feature=feature)

    print(f'通过机器学习方法得到的预测标签={y_pred}')

    # 根据我设定的规则，为验证集设置标签
    y_true = get_labels(data, feature)
    print(f'真实标签={y_true}')

    # 调用精度方法
    acc = acc_metric(y_true=y_true, y_pred=y_pred)
    # acc = mcc_metric(y_true=y_true, y_pred=y_pred)
    # acc = f1_metric(y_true=y_true, y_pred=y_pred)
    # acc = recall_metric(y_true=y_true, y_pred=y_pred)
    print(f'精度={acc}')


if __name__ == '__main__':
    main()

