import random
from functools import wraps

"""
采用修饰器技术对作业1随机数据结构生成函数进行修饰，实现所有生成随机数的四种机器学习方法（SVM,RF,CNN,RNN）操作，
四种精度指标（ACC,MCC,F1,RECALL）操作，以及相应的调用示例。
主要考察点是带参数修饰器的使用，具体要求如下：
1. 修饰器类型不限，可以是函数修饰器或类修饰器；
2. 实现两个修饰器，通过修饰器参数（*args）实现机器学习方法和验证指标操作的任意组合；
"""


def ml_operation(*args):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            for op in args:
                if op == 'SVM':
                    print('SVM operation')
                elif op == 'RF':
                    print('RF operation')
                elif op == 'CNN':
                    print('CNN operation')
                elif op == 'RNN':
                    print('RNN operation')
            return result
        return wrapper
    return decorator


def accuracy_operation(*args):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            for op in args:
                if op == 'ACC':
                    print('ACC operation')
                elif op == 'MCC':
                    print('MCC operation')
                elif op == 'F1':
                    print('F1 operation')
                elif op == 'RECALL':
                    print('RECALL operation')
            return result
        return wrapper
    return decorator


# 随机数据结构生成函数
@ml_operation('SVM', 'RF')
@accuracy_operation('ACC', 'F1')
def dataSampling(**kwargs):
    res = []
    for data_type, data_len in kwargs.items():
        if data_type == "int":
            res.extend(random.sample(range(1, 100), data_len))
        elif data_type == "float":
            res.extend([round(random.uniform(0, 100), 3) for _ in range(data_len)])
        elif data_type == "str":
            res.extend([''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=7)) for _ in range(data_len)])
    return res


# 调用示例
results = dataSampling(int=5, float=4, str=3)
print(results)
