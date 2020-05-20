import torch.utils.data
from torch.autograd import Variable
from torchvision import datasets, transforms

from models import VAE, loss_function
from utils import load_params

BATCH_SIZE = 8
DATASET_PATH = "../data"

# 准备数据
testset = datasets.MNIST(DATASET_PATH, train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

model = VAE()
load_params(model, "VAE.pkl")


# 测试模型的效果
def test():
    model.eval()
    test_loss = 0
    # 加载测试数据
    for data, _ in test_loader:
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        # 计算损失
        test_loss += loss_function(recon_batch, data, mu, logvar).data
    # 求得平均损失
    test_loss /= len(test_loader.dataset)
    print('====> test loss: {:.4f}'.format(test_loss))


def show_test_result(n_img=4):
    model.eval()
    test_data = None
    test_label = []
    # 加载测试数据
    for data, label in test_loader:
        # 防止计算梯度
        data = Variable(data, volatile=True)
        if len(data) > n_img:
            if test_data is None:
                test_data = data[:n_img]
            else:
                test_data = torch.cat(test_data, data[:n_img])
            test_label.extend(label[:n_img].numpy())
            break
        else:
            n_img -= len(data)
            if test_data is None:
                test_data = data
            else:
                test_data = torch.cat(test_data, data)
            test_label.extend(label.numpy())
    # 将测试数据放入model
    test_data = Variable(test_data, volatile=True)
    results, _, _ = model(test_data)
    # 输出的结果是tensor
    # 查看结果
    from PIL import Image
    import numpy as np
    results = results.detach().numpy()
    results = results * 255
    # 如果直接转换成np.int8会导致结果出现负数
    results = results.astype(np.uint8)
    for index, result in enumerate(results):
        result = result.reshape((28, 28))
        from matplotlib import pyplot as plt
        img = Image.fromarray(result * 255)
        plt.subplot(1, n_img, index + 1)
        plt.xlabel(test_label[index])
        plt.imshow(img)

        # %matplotlib inline
        # 百分号开头的都是 Magic Function, 可以自行了解


if __name__ == '__main__':
    with torch.no_grad():
        test()
        show_test_result()
