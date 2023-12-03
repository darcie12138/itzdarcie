# Equal Distribution Algorithm


#### 定义函数

```
from itertools import count
import numpy as np
import time

def avg_per_group(data):
    sum_per_group = data.sum(axis=1)   # 每行总和
    count_per_group = (data!=0).sum(axis=1)  # 每行非零数据个数
    return sum_per_group / count_per_group

def balance(data, shape, diff_expected=0.1, max_exchange=1000000):
    rows, cols, concav = shape
    # 每行平均值=每行total/每行非零数据个数
    def avg_per_group(data):
        sum_per_group = data.sum(axis=1)   # 每行总和
        count_per_group = (data!=0).sum(axis=1)  # 每行非零数据个数
        return sum_per_group / count_per_group

    begin_time = time.time() 
    total_exchange = 0 # 交换次数
    for epoch in count(1, step=1):             # 从1开始，步长为1,epoch表示轮次
        diff_begin = avg_per_group(data).ptp()       # 极差,ptp()函数表示最大值与最小值的差
        if diff_begin < diff_expected:
            print('='*80)
            print('极差已达标，成功优化!')
            break

        row1, row2 = avg_per_group(data).argmax(), avg_per_group(data).argmin()   # 最大行，最小行
        bak = avg_per_group(data)[[row1, row2]]                             # 备份最大行，最小行
        valid_col1 = cols - (row1 >= concav)  # 最大行有效数据个数
        valid_col2 = cols - (row2 >= concav)  # 最小行有效数据个数
        diff_end = diff_begin
        for i in range(max_exchange):
            col1 = np.random.randint(0, valid_col1)   # 随机列1
            col2 = np.random.randint(0, valid_col2)   # 随机列2
            data[row1, col1], data[row2, col2] = data[row2, col2], data[row1, col1]  # 交换

            diff_end = avg_per_group(data).ptp()   # 交换后的极差
            if abs(diff_end) < diff_begin:   # 交换后的极差小于交换前的极差,abs()函数表示取绝对值
                temp = avg_per_group(data)[row1] + avg_per_group(data)[row2] 
                avg_per_group(data)[row1] = (temp + diff_end) / 2
                avg_per_group(data)[row2] = (temp - diff_end) / 2
                if epoch%1000==0 or i>100000:
                    print(f'轮次：{epoch:<8}， 交换次数：{i+1:<10}， 初始差距{diff_begin:<8}， 结束差距：{abs(diff_end):<8}')
                total_exchange += i+1
                break
        else:
            avg_per_group(data)[[row1, row2]] = bak  
            print('超过最大允许交换数，未达到优化目标!')
            total_exchange += max_exchange
            break

    print('最终极差为：', avg_per_group(data).ptp())
    elapsed_time = time.time()-begin_time
    print(f'{rows}行{cols}列的矩阵，优化{epoch-1}轮')
```

#### 平均分配

```python
import numpy as np
import pandas as pd
import math

MQL = ["Abandoner", "NS-psn", "NS-biz", "New Service", "New Biller", "Fast Mover", "Others"]
MQL_cr = [0.06, 0.09, 0.139, 0.103, 0.26, 0.172, 0.1]
MQL_counts = [7, 3, 5, 24, 0, 0, 0]  # replace with actual counts
DGR = ["Zhong Fachao", "Huang Jiaqi",  "Fang Jun", "Zhang Rebecca", "Ma Haocheng", "Yang Song", "Sun Weiqiu","Wang Yang", "Zhao Hongyu"]   # "Huang Yudi", "Zhang Jinzhen", "Liang Chuang",
m = len(DGR)
N = sum(MQL_counts)
print("MQL总数：",N)
# Create a 2D matrix with default values as 0
cols = math.ceil(N / m)             
matrix = np.zeros((m, cols))

# Fill the matrix with the sweetness of candies
MQL_index = 0  # MQL_index表示MQL_cr的索引
ct = 1
for j in range(cols):   # j表示列
    for i in range(m):         # i表示行
        if ct <= N:            # ct表示已经分配的MQL数量
            while MQL_counts[MQL_index] == 0 or MQL_cr[MQL_index] == 0:
                MQL_index += 1
            if MQL_counts[MQL_index] > 0:
                matrix[i][j] = MQL_cr[MQL_index]
                MQL_counts[MQL_index] -= 1
                ct += 1
            else:
                MQL_index += 1
                if MQL_index < len(MQL_counts):
                    matrix[i][j] = MQL_cr[MQL_index]
                    MQL_counts[MQL_index] -= 1
                    ct += 1
                else:
                    break
        else:
            break
if m * math.ceil(N/m) - N == 0:
    concave = 0
else:
    concave = m -(m * math.ceil(N/m) - N)

print("DGR人数：",m)

# print(candy_per_row)  矩阵的列数，即每个DGR被分到的MQL数量
# print(concave)
balance(
    data = matrix,
    shape = (m, cols, concave),
    diff_expected = 0.005
)

results = np.zeros((m, len(MQL_cr)))
# 初始化输出表格
for i in range(len(matrix)):
    results[i] = [0] * len(MQL_cr)
# 遍历矩阵并更新results表格
for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        if matrix[i][j] in MQL_cr:
            column_index = MQL_cr.index(matrix[i][j])
            results[i][column_index] += 1

# 计算每行的总和和平均值
row_sums = [sum(row) for row in results]
row_averages = [sum(row) / len(row) for row in results]


# 输出结果
for i in range(len(results)):
    # CR保留2位小数
    print(DGR[i],results[i],"Total #MQL=", sum(results[i]),"," ,"CR =", round(sum(matrix[i])/sum(results[i]), 4))

# 转换为整数格式
results = results.astype(int)

# 将results转换为DataFrame
df = pd.DataFrame(results, columns=MQL,index = DGR)

# 增加两列
df['Total #MQL'] = df.apply(lambda x: x.sum(), axis=1)
for i in range(len(results)):
    df['CR'] = df.apply(lambda x: round(sum(matrix[i])/sum(results[i]), 4))


# 输出表格
print(df)

# 输出每种MQL的个数
print(df.sum(axis=0))
```

<!--more-->


