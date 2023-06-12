import random
"""
采用类工厂设计模式实现作业2需求，以及相应的调用示例，主要考察点是应用创建模式搭建科学实验基本框架。
"""


class DataSampler:
    def __init__(self):
        pass

    def dataSampling(self, **kwargs):
        res = []
        for data_type, data_len in kwargs.items():
            if data_type == "int":
                res.extend(random.sample(range(1, 100), data_len))
            elif data_type == "float":
                res.extend([round(random.uniform(0, 100), 3) for _ in range(data_len)])
            elif data_type == "str":
                res.extend([''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(5, 10))) for _ in
                            range(data_len)])
            else:
                print(f'unsupported data type: {data_type}')
        return res


class Factory:
    def __init__(self, models=None, metrics=None):
        self.models = models or []
        self.metrics = metrics or []

    def create_data_sampler(self):
        print("机器学习方法模型:")
        for model in self.models:
            # print(f"  - {model}")
            if model == 'SVM':
                print('执行了SVM操作...')
            elif model == 'RF':
                print('执行了RF操作...')
            elif model == 'CNN':
                print('执行了CNN操作...')
            elif model == 'RNN':
                print('执行了RNN操作...')

        print("\n精度指标操作:")
        for metric in self.metrics:
            # print(f"  - {metric}")
            if metric == 'ACC':
                print('执行了ACC操作...')
            elif metric == 'MCC':
                print('执行了MCC操作...')
            elif metric == 'F1':
                print('执行了F1操作...')
            elif metric == 'RECALL':
                print('执行了RECALL操作...')

        return DataSampler()


# 调用示例
def main():
    models = ["SVM", "RF", "CNN", "RNN"]
    metrics = ["ACC", "MCC", "F1", "RECALL"]

    factory = Factory(models=models, metrics=metrics)
    data_sampler = factory.create_data_sampler()
    results = data_sampler.dataSampling(int=5, float=4, str=3)
    print(results)


if __name__ == '__main__':
    main()

