import numpy as np
import pandas as pd

temp_path = "test.csv"
# 创建数据
data = np.random.randn(6, 4)
# df = pd.DataFrame(data, columns=list('ABCD'), index=[1, 2, 'a', 'b', '2006-10-1', '第六行'])
df = pd.DataFrame(data, columns=list('ABCD'), index=range(6))

print(df)
print("=" * 20)
# 读取数据
print(df.A)
print(df[['A']])
# 读取多列
print(df[['A', 'C', 'D']])
# 使用列号来读取
print(df.iloc[:, 1])

# 按行来读取
print(df.loc[0])
print(df.iloc[0])
# .iloc根据行号索引数据，行号是固定不变的，不受索引变化的影响
# 如果df的索引是默认值，则.loc和.iloc的用法没有区别，因为此时行号和行标签相同。
# .ix 既可以通过行标签检索也可以根据行号检索。

# 按单元格读取
print(df['A'][1])
print(df.A[1])
# (1)读取一个单元格：df.loc[row][col]或df.loc[row,col]
# (2)读取一行多列：
# df.loc[row][[col1,col2]]
# df.loc[row][firstCol:endCol]
print(df.loc[1][['A', 'B']])

# 保存数据
df.to_csv('submission.csv', index=False)


