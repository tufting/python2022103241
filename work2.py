import random

"""
采用修饰器技术对作业1随机数据结构生成函数进行修饰，实现所有生成随机数的四种机器学习方法（SVM,RF,CNN,RNN）操作，四种精度指标（ACC,MCC,F1,RECALL）操作，以及相应的调用示例。主要考察点是带参数修饰器的使用，具体要求如下：
1. 修饰器类型不限，可以是函数修饰器或类修饰器；
2. 实现两个修饰器，通过修饰器参数（*args）实现机器学习方法和验证指标操作的任意组合；
"""

# 随机数据结构生成函数
def dataSampling(**kwargs):
    result = []
    for data_type, data_size in kwargs.items():
        if data_type == "int":
            result.extend(random.sample(range(1, 100), data_size))
        elif data_type == "float":
            result.extend([random.uniform(0, 1) for _ in range(data_size)])
        elif data_type == "str":
            result.extend([''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5)) for _ in range(data_size)])
    return result

# 机器学习方法修饰器
def ml_method(func):
    def wrapper(*args):
        print("Applying machine learning method:", func.__name__)
        result = func(*args)
        return result
    return wrapper

# 验证指标操作修饰器
def performance_metric(func):
    def wrapper(*args):
        print("Calculating performance metric:", func.__name__)
        result = func(*args)
        return result
    return wrapper

# 调用示例
@ml_method
@performance_metric
def svm(data):
    # SVM操作
    print("SVM")
    # 返回结果
    return "SVM result"

@ml_method
@performance_metric
def rf(data):
    # RF操作
    print("RF")
    # 返回结果
    return "RF result"

# 调用示例
random_data = dataSampling(int=5, float=3, str=4)
svm_result = svm(random_data)
rf_result = rf(random_data)
print(svm_result)
print(rf_result)
