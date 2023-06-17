import random

"""
采用修饰器技术对作业1随机数据结构生成函数进行修饰，实现所有生成随机数的四种机器学习方法（SVM,RF,CNN,RNN）操作，
四种精度指标（ACC,MCC,F1,RECALL）操作，以及相应的调用示例。
主要考察点是带参数修饰器的使用，具体要求如下：
1. 修饰器类型不限，可以是函数修饰器或类修饰器；
2. 实现两个修饰器，通过修饰器参数（*args）实现机器学习方法和验证指标操作的任意组合；
"""


# 机器学习方法修饰器
def ml_decorator(*models):
    def outer_wrapper(func):
        def inner_wrapper(**kwargs):
            print('机器学习方法模型:')
            for model in models:
                # print(f'执行了{model}操作...')
                if model == 'SVM':
                    print('执行了SVM操作...')
                elif model == 'RF':
                    print('执行了RF操作...')
                elif model == 'CNN':
                    print('执行了CNN操作...')
                elif model == 'RNN':
                    print('执行了RNN操作...')
            result = func(**kwargs)
            return result
        return inner_wrapper
    return outer_wrapper


# 精度指标修饰器
def metrics_decorator(*metrics):
    def outer_wrapper(func):
        def inner_wrapper(**kwargs):
            print('\n精度指标操作:')
            for metric in metrics:
                # print(f'执行了{metric}操作...')
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
@ml_decorator('SVM', 'RF', 'CNN', 'RNN')
@metrics_decorator('ACC', 'MCC', 'F1', 'RECALL')
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

    print('当前获取的随机列表为{}, 我想要的维度是{}.'.format(res, new_shape))
    return reshape(res, new_shape)


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


# 调用示例
def main():
    results = dataSampling(shape=(3, 2, 2), type=(int, str, float, int))
    print(results)


if __name__ == '__main__':
    main()

