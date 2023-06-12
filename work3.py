import random
"""
采用类工厂设计模式实现作业2需求，以及相应的调用示例，主要考察点是应用创建模式搭建科学实验基本框架。
"""


# 随机数据结构生成类
class DataGenerator:
    @staticmethod
    def generate(**kwargs):
        result = []
        for data_type, data_size in kwargs.items():
            if data_type == "int":
                result.extend(random.sample(range(1, 100), data_size))
            elif data_type == "float":
                result.extend([random.uniform(0, 1) for _ in range(data_size)])
            elif data_type == "str":
                result.extend([''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5)) for _ in range(data_size)])
        return result

# 机器学习方法类
class MachineLearning:
    def __init__(self, method):
        self.method = method

    def apply_method(self, data):
        print("Applying machine learning method:", self.method)
        # 实际的机器学习方法操作
        # ...
        return self.method + " result"

# 验证指标类
class PerformanceMetric:
    def __init__(self, metric):
        self.metric = metric

    def calculate_metric(self, data):
        print("Calculating performance metric:", self.metric)
        # 实际的验证指标操作
        # ...
        return self.metric + " result"

# 类工厂
class Factory:
    @staticmethod
    def create_ml_object(method):
        return MachineLearning(method)

    @staticmethod
    def create_metric_object(metric):
        return PerformanceMetric(metric)

# 调用示例
data = DataGenerator.generate(int=5, float=3, str=4)

ml_obj = Factory.create_ml_object("SVM")
svm_result = ml_obj.apply_method(data)
print(svm_result)

metric_obj = Factory.create_metric_object("ACC")
acc_result = metric_obj.calculate_metric(data)
print(acc_result)