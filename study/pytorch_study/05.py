import torch
import torch.utils.data as Data

# 使用DataLoader 来设置batchsize
torch.manual_seed(1)  # 为CPU设置种子用于生成随机数，以使得结果是确定的

BATCH_SIZE = 8
x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

# create TensorDataset
torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batchsize
    shuffle=True,  # 是否打乱数据
    num_workers=2  # 使用多线程来读数据
)

for i in range(30):
    for step, (batch_x, batch_y) in enumerate(loader):
        print('Epoch:', i, '   |   Step:', step, '   |   batch x:',
              batch_x.numpy(), "   |   batch y:", batch_y.numpy())
