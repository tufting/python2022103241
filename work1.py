import random

"""
实现随机数据结构生成封装函数dataSampling(**kwargs)，以及相应的调用示例，主要考察点是关键字参数的使用，具体要求如下：
1. 采用关键字参数作为随机数据结构及数量的输入；
2. 在不修改代码的前提下，根据kwargs定义内容实现任意数量、任意维度、一层深度的随机数据结构生成；
3. 其中随机数涵盖int，float和str三种类型。
"""

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


results = dataSampling(int=5, float=4, str=3)
print(results)

