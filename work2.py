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
    res = []
    for data_type, data_len in kwargs.items():
        if data_type == "int":
            res.extend(random.sample(range(1, 100), data_len))
        elif data_type == "float":
            res.extend([round(random.uniform(0, 100), 3) for _ in range(data_len)])
        elif data_type == "str":
            res.extend([''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(5, 10))) for _ in range(data_len)])
        else:
            print(f'unsupported data type: {data_type}')
    return res


# 调用示例
def main():
    results = dataSampling(int=5, float=4, str=3)
    print(results)


if __name__ == '__main__':
    main()

