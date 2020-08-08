import torch
from torch import nn
from torch.autograd import Variable


class CNNAutoEncoder(nn.Module):
    def __init__(self):
        super(CNNAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class LinearAutoEncoder(nn.Module):
    def __init__(self):
        super(LinearAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 10),
            nn.Sigmoid(),
            nn.Linear(10, 2),
        )
        self.decoder = nn.Sequential(
            # nn.Sigmoid(),
            nn.Linear(2, 10),
            nn.Sigmoid(),
            nn.Linear(10, 3)
        )

    def forward(self, x):
        encoder_output = self.encoder(x)
        output = self.decoder(encoder_output)
        return output, encoder_output


def train(num_epoch=float("inf"), min_loss=0.00001, max_try_times=None):
    last_loss = float("inf")
    try_times = 0
    epoch = 0
    while True:
        train_loss = train_one_epoch(epoch)
        if train_loss > last_loss:
            try_times += 1
        else:
            try_times = 0
        last_loss = train_loss
        if try_times is not None and try_times == max_try_times:
            print("loss don't decrease in {} epoch".format(max_try_times))
            break
        if train_loss < min_loss:
            break
        if num_epoch < epoch:
            break
        epoch += 1
    # save model
    torch.save(net.state_dict(), model_save_path)


def train_one_epoch(epoch):
    train_loss = 0
    net.train()
    global data
    for sample, label in train_data:
        sample = Variable(torch.from_numpy(sample)).double()
        label = Variable(torch.from_numpy(label)).double()
        optimizer.zero_grad()
        out = net(sample)[0]
        loss = loss_function(out, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.data
    train_loss = train_loss / len(train_data)
    print("epoch:{} \t train loss \t {}".format(epoch, train_loss))
    return train_loss


def test():
    all_loss = 0
    net.load_state_dict(torch.load(model_save_path))
    net.eval()
    hidden_values = []
    for i, (data) in enumerate(train_data):
        data = Variable(torch.from_numpy(data)).double()
        gt = Variable(torch.from_numpy(data)).double()
        optimizer.zero_grad()
        out, hidden_ouput = net(data)
        batch_loss = loss_function(out, gt)
        all_loss += batch_loss
        hidden_values.append(hidden_ouput.data.numpy().tolist())
        # print("m:{},n:{} output of the hidden layer {}".format(m, n, hidden_ouput.data))
    print(M_N)
    print(hidden_values)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    from sklearn.model_selection import cross_val_score

    def rmse_cv(model, x, y):
        return np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=5))

    hidden_values0 = [d[0] for d in hidden_values]
    hidden_values1 = [d[1] for d in hidden_values]
    from scipy.stats import pearsonr
    M = [d[1] for d in M_N]
    print("Lower noise", pearsonr(M, hidden_values1))

    score = rmse_cv(model, M_N, hidden_values0)
    print("{}: {:6f}, {:6f}".format("LR", score.mean(), score.std()))
    score = rmse_cv(model, M_N, hidden_values1)
    print("{}: {:6f}, {:6f}".format("LR", score.mean(), score.std()))

    print("train loss \t {}".format(all_loss / len(test_data)))


def set_seed(random_state):
    import torch
    import numpy as np
    import random
    if random_state is not None:
        torch.manual_seed(random_state)  # cpu
        torch.cuda.manual_seed(random_state)  # gpu
        torch.cuda.manual_seed_all(random_state)
        np.random.seed(random_state)  # numpy
        random.seed(random_state)  # random and transforms
        torch.backends.cudnn.deterministic = True  # cudnn


if __name__ == '__main__':
    import numpy as np

    set_seed(0)
    model_save_path = "AutoEncoder1.pkl"
    net = LinearAutoEncoder().double()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.03, weight_decay=0.0001)

    # prepare data
    train_data = []
    last_loss = 0
    M_N = []
    for i in range(100):
        m = np.random.rand() * np.pi
        n = np.random.rand() * np.pi / 2
        M_N.append([m, n])
        x = np.cos(m) * np.sin(n)
        y = np.sin(m) * np.sin(n)
        z = np.cos(n)
        train_data.append((np.array([x, y, z], dtype=np.float32), np.array([x, y, z], dtype=np.float32)))
    print(train_data)
    # [(array([-0.13771854,  0.8910034 ,  0.43260443], dtype=float32), array([-0.13771854,  0.8910034 ,  0.43260443], dtype=float32)), (array([-0.23958386,  0.7161484 ,  0.65553874], dtype=float32), array([-0.23958386,  0.7161484 ,  0.65553874], dtype=float32)), (array([0.20174214, 0.82494247, 0.52798676], dtype=float32), array([0.20174214, 0.82494247, 0.52798676], dtype=float32)), (array([0.19201311, 0.96669924, 0.16918488], dtype=float32), array([0.19201311, 0.96669924, 0.16918488], dtype=float32)), (array([-0.56285876,  0.06453473,  0.8240299 ], dtype=float32), array([-0.56285876,  0.06453473,  0.8240299 ], dtype=float32)), (array([-0.5859435 ,  0.4494396 ,  0.67429537], dtype=float32), array([-0.5859435 ,  0.4494396 ,  0.67429537], dtype=float32)), (array([-0.21069671,  0.9705719 ,  0.11660665], dtype=float32), array([-0.21069671,  0.9705719 ,  0.11660665], dtype=float32)), (array([0.13305213, 0.03019571, 0.9906489 ], dtype=float32), array([0.13305213, 0.03019571, 0.9906489 ], dtype=float32)), (array([0.96368784, 0.06129395, 0.25990143], dtype=float32), array([0.96368784, 0.06129395, 0.25990143], dtype=float32)), (array([-0.75088   ,  0.62854123,  0.20276861], dtype=float32), array([-0.75088   ,  0.62854123,  0.20276861], dtype=float32)), (array([-0.9485033 ,  0.06380931,  0.31027377], dtype=float32), array([-0.9485033 ,  0.06380931,  0.31027377], dtype=float32)), (array([0.11361801, 0.9342788 , 0.3379557 ], dtype=float32), array([0.11361801, 0.9342788 , 0.3379557 ], dtype=float32)), (array([0.78664774, 0.30653343, 0.5359315 ], dtype=float32), array([0.78664774, 0.30653343, 0.5359315 ], dtype=float32)), (array([0.8968932 , 0.43364447, 0.08680448], dtype=float32), array([0.8968932 , 0.43364447, 0.08680448], dtype=float32)), (array([-0.04158014,  0.6048326 ,  0.7952664 ], dtype=float32), array([-0.04158014,  0.6048326 ,  0.7952664 ], dtype=float32)), (array([0.63210136, 0.69272506, 0.3472461 ], dtype=float32), array([0.63210136, 0.69272506, 0.3472461 ], dtype=float32)), (array([0.10695912, 0.7715111 , 0.6271606 ], dtype=float32), array([0.10695912, 0.7715111 , 0.6271606 ], dtype=float32)), (array([0.8235503 , 0.04867063, 0.5651514 ], dtype=float32), array([0.8235503 , 0.04867063, 0.5651514 ], dtype=float32)), (array([-0.2843439,  0.7737729,  0.5660601], dtype=float32), array([-0.2843439,  0.7737729,  0.5660601], dtype=float32)), (array([-0.86401117,  0.1542982 ,  0.47924608], dtype=float32), array([-0.86401117,  0.1542982 ,  0.47924608], dtype=float32)), (array([0.27075577, 0.57308394, 0.77347666], dtype=float32), array([0.27075577, 0.57308394, 0.77347666], dtype=float32)), (array([-0.05495249,  0.07683155,  0.9955286 ], dtype=float32), array([-0.05495249,  0.07683155,  0.9955286 ], dtype=float32)), (array([-0.43480033,  0.7525499 ,  0.49458808], dtype=float32), array([-0.43480033,  0.7525499 ,  0.49458808], dtype=float32)), (array([0.15877993, 0.12346827, 0.9795634 ], dtype=float32), array([0.15877993, 0.12346827, 0.9795634 ], dtype=float32)), (array([0.29626966, 0.45235285, 0.84119034], dtype=float32), array([0.29626966, 0.45235285, 0.84119034], dtype=float32)), (array([-0.13906367,  0.62033355,  0.7719116 ], dtype=float32), array([-0.13906367,  0.62033355,  0.7719116 ], dtype=float32)), (array([-0.15949965,  0.00582826,  0.98718077], dtype=float32), array([-0.15949965,  0.00582826,  0.98718077], dtype=float32)), (array([0.19861837, 0.15294467, 0.96806955], dtype=float32), array([0.19861837, 0.15294467, 0.96806955], dtype=float32)), (array([-0.17926368,  0.34349096,  0.92188853], dtype=float32), array([-0.17926368,  0.34349096,  0.92188853], dtype=float32)), (array([0.03957067, 0.37248313, 0.92719495], dtype=float32), array([0.03957067, 0.37248313, 0.92719495], dtype=float32)), (array([0.15143952, 0.08261732, 0.9850078 ], dtype=float32), array([0.15143952, 0.08261732, 0.9850078 ], dtype=float32)), (array([-0.10156602,  0.18990242,  0.97653544], dtype=float32), array([-0.10156602,  0.18990242,  0.97653544], dtype=float32)), (array([0.44624254, 0.31695023, 0.8369051 ], dtype=float32), array([0.44624254, 0.31695023, 0.8369051 ], dtype=float32)), (array([-0.12853688,  0.08101049,  0.9883904 ], dtype=float32), array([-0.12853688,  0.08101049,  0.9883904 ], dtype=float32)), (array([-0.13130714,  0.07329462,  0.9886285 ], dtype=float32), array([-0.13130714,  0.07329462,  0.9886285 ], dtype=float32)), (array([-0.66960865,  0.04961123,  0.7410553 ], dtype=float32), array([-0.66960865,  0.04961123,  0.7410553 ], dtype=float32)), (array([-0.8113004 ,  0.05933623,  0.5816106 ], dtype=float32), array([-0.8113004 ,  0.05933623,  0.5816106 ], dtype=float32)), (array([-0.04200754,  0.0449414 ,  0.998106  ], dtype=float32), array([-0.04200754,  0.0449414 ,  0.998106  ], dtype=float32)), (array([0.11835478, 0.14566281, 0.98222935], dtype=float32), array([0.11835478, 0.14566281, 0.98222935], dtype=float32)), (array([0.11079678, 0.14867364, 0.98265976], dtype=float32), array([0.11079678, 0.14867364, 0.98265976], dtype=float32)), (array([0.3278174 , 0.50939465, 0.7956462 ], dtype=float32), array([0.3278174 , 0.50939465, 0.7956462 ], dtype=float32)), (array([0.8676541 , 0.17726044, 0.46449447], dtype=float32), array([0.8676541 , 0.17726044, 0.46449447], dtype=float32)), (array([-0.08410294,  0.3960722 ,  0.9143596 ], dtype=float32), array([-0.08410294,  0.3960722 ,  0.9143596 ], dtype=float32)), (array([-0.01072867,  0.14663452,  0.9891326 ], dtype=float32), array([-0.01072867,  0.14663452,  0.9891326 ], dtype=float32)), (array([-0.23487961,  0.965685  ,  0.1108331 ], dtype=float32), array([-0.23487961,  0.965685  ,  0.1108331 ], dtype=float32)), (array([0.46763715, 0.7296071 , 0.49898794], dtype=float32), array([0.46763715, 0.7296071 , 0.49898794], dtype=float32)), (array([0.8261037 , 0.36304036, 0.4309923 ], dtype=float32), array([0.8261037 , 0.36304036, 0.4309923 ], dtype=float32)), (array([0.17436205, 0.22392225, 0.95888305], dtype=float32), array([0.17436205, 0.22392225, 0.95888305], dtype=float32)), (array([-0.00847769,  0.03042039,  0.9995012 ], dtype=float32), array([-0.00847769,  0.03042039,  0.9995012 ], dtype=float32)), (array([-0.00633593,  0.00377559,  0.9999728 ], dtype=float32), array([-0.00633593,  0.00377559,  0.9999728 ], dtype=float32)), (array([-0.21811792,  0.3489671 ,  0.9113981 ], dtype=float32), array([-0.21811792,  0.3489671 ,  0.9113981 ], dtype=float32)), (array([-0.6722757 ,  0.73791724,  0.05935918], dtype=float32), array([-0.6722757 ,  0.73791724,  0.05935918], dtype=float32)), (array([0.5582719 , 0.55391526, 0.6176653 ], dtype=float32), array([0.5582719 , 0.55391526, 0.6176653 ], dtype=float32)), (array([-0.2231655,  0.7501454,  0.6224781], dtype=float32), array([-0.2231655,  0.7501454,  0.6224781], dtype=float32)), (array([0.7622019 , 0.64307815, 0.07415355], dtype=float32), array([0.7622019 , 0.64307815, 0.07415355], dtype=float32)), (array([0.16055879, 0.9576715 , 0.23892699], dtype=float32), array([0.16055879, 0.9576715 , 0.23892699], dtype=float32)), (array([-0.2641418 ,  0.36481354,  0.8928271 ], dtype=float32), array([-0.2641418 ,  0.36481354,  0.8928271 ], dtype=float32)), (array([-0.4863434 ,  0.32210383,  0.812231  ], dtype=float32), array([-0.4863434 ,  0.32210383,  0.812231  ], dtype=float32)), (array([-0.73681074,  0.2887741 ,  0.611326  ], dtype=float32), array([-0.73681074,  0.2887741 ,  0.611326  ], dtype=float32)), (array([-0.8251932 ,  0.32152426,  0.46441174], dtype=float32), array([-0.8251932 ,  0.32152426,  0.46441174], dtype=float32)), (array([-0.46061376,  0.53843784,  0.70563424], dtype=float32), array([-0.46061376,  0.53843784,  0.70563424], dtype=float32)), (array([-0.83961487,  0.11658006,  0.53052425], dtype=float32), array([-0.83961487,  0.11658006,  0.53052425], dtype=float32)), (array([0.19307858, 0.7916744 , 0.57963127], dtype=float32), array([0.19307858, 0.7916744 , 0.57963127], dtype=float32)), (array([0.45536417, 0.02749051, 0.8898808 ], dtype=float32), array([0.45536417, 0.02749051, 0.8898808 ], dtype=float32)), (array([-0.21220525,  0.38550192,  0.89797395], dtype=float32), array([-0.21220525,  0.38550192,  0.89797395], dtype=float32)), (array([-0.22599094,  0.58135164,  0.78163826], dtype=float32), array([-0.22599094,  0.58135164,  0.78163826], dtype=float32)), (array([0.41129866, 0.1864464 , 0.8922282 ], dtype=float32), array([0.41129866, 0.1864464 , 0.8922282 ], dtype=float32)), (array([-0.17453907,  0.7812475 ,  0.59932333], dtype=float32), array([-0.17453907,  0.7812475 ,  0.59932333], dtype=float32)), (array([-0.19789232,  0.832047  ,  0.51820505], dtype=float32), array([-0.19789232,  0.832047  ,  0.51820505], dtype=float32)), (array([-0.288328  ,  0.55675024,  0.7790354 ], dtype=float32), array([-0.288328  ,  0.55675024,  0.7790354 ], dtype=float32)), (array([-0.5172432 ,  0.1742888 ,  0.83790386], dtype=float32), array([-0.5172432 ,  0.1742888 ,  0.83790386], dtype=float32)), (array([0.1972488, 0.9656853, 0.1689521], dtype=float32), array([0.1972488, 0.9656853, 0.1689521], dtype=float32)), (array([-0.7331545 ,  0.5111719 ,  0.44853964], dtype=float32), array([-0.7331545 ,  0.5111719 ,  0.44853964], dtype=float32)), (array([0.94324124, 0.30722114, 0.12613949], dtype=float32), array([0.94324124, 0.30722114, 0.12613949], dtype=float32)), (array([-0.6233797 ,  0.7819172 ,  0.00181112], dtype=float32), array([-0.6233797 ,  0.7819172 ,  0.00181112], dtype=float32)), (array([0.872727  , 0.44277298, 0.20566884], dtype=float32), array([0.872727  , 0.44277298, 0.20566884], dtype=float32)), (array([0.71819526, 0.40218773, 0.56783855], dtype=float32), array([0.71819526, 0.40218773, 0.56783855], dtype=float32)), (array([0.89904577, 0.3684982 , 0.23648643], dtype=float32), array([0.89904577, 0.3684982 , 0.23648643], dtype=float32)), (array([-0.6410343 ,  0.44358504,  0.62634444], dtype=float32), array([-0.6410343 ,  0.44358504,  0.62634444], dtype=float32)), (array([0.03117225, 0.10385637, 0.99410367], dtype=float32), array([0.03117225, 0.10385637, 0.99410367], dtype=float32)), (array([-0.37993306,  0.5319161 ,  0.75678015], dtype=float32), array([-0.37993306,  0.5319161 ,  0.75678015], dtype=float32)), (array([-0.6282891 ,  0.7495623 ,  0.20834856], dtype=float32), array([-0.6282891 ,  0.7495623 ,  0.20834856], dtype=float32)), (array([-0.9715776 ,  0.07486337,  0.2245718 ], dtype=float32), array([-0.9715776 ,  0.07486337,  0.2245718 ], dtype=float32)), (array([0.5354349 , 0.01971338, 0.8443464 ], dtype=float32), array([0.5354349 , 0.01971338, 0.8443464 ], dtype=float32)), (array([-0.17612877,  0.1997908 ,  0.96387875], dtype=float32), array([-0.17612877,  0.1997908 ,  0.96387875], dtype=float32)), (array([-0.00562996,  0.08506421,  0.9963596 ], dtype=float32), array([-0.00562996,  0.08506421,  0.9963596 ], dtype=float32)), (array([0.02353438, 0.01709834, 0.9995768 ], dtype=float32), array([0.02353438, 0.01709834, 0.9995768 ], dtype=float32)), (array([-0.27466822,  0.20798938,  0.93877465], dtype=float32), array([-0.27466822,  0.20798938,  0.93877465], dtype=float32)), (array([0.46397662, 0.8786454 , 0.1127295 ], dtype=float32), array([0.46397662, 0.8786454 , 0.1127295 ], dtype=float32)), (array([-0.02994239,  0.04003269,  0.9987496 ], dtype=float32), array([-0.02994239,  0.04003269,  0.9987496 ], dtype=float32)), (array([0.7199542, 0.4097383, 0.5601612], dtype=float32), array([0.7199542, 0.4097383, 0.5601612], dtype=float32)), (array([-0.08770114,  0.3543535 ,  0.93098986], dtype=float32), array([-0.08770114,  0.3543535 ,  0.93098986], dtype=float32)), (array([-0.8042291 ,  0.16861995,  0.5698973 ], dtype=float32), array([-0.8042291 ,  0.16861995,  0.5698973 ], dtype=float32)), (array([-0.08932336,  0.79459494,  0.6005333 ], dtype=float32), array([-0.08932336,  0.79459494,  0.6005333 ], dtype=float32)), (array([-0.311367 ,  0.3529037,  0.8823319], dtype=float32), array([-0.311367 ,  0.3529037,  0.8823319], dtype=float32)), (array([0.1017431 , 0.3072791 , 0.94616485], dtype=float32), array([0.1017431 , 0.3072791 , 0.94616485], dtype=float32)), (array([0.8305629 , 0.55004495, 0.08726849], dtype=float32), array([0.8305629 , 0.55004495, 0.08726849], dtype=float32)), (array([-0.47602233,  0.50834805,  0.71762455], dtype=float32), array([-0.47602233,  0.50834805,  0.71762455], dtype=float32)), (array([0.29387048, 0.25486967, 0.92123914], dtype=float32), array([0.29387048, 0.25486967, 0.92123914], dtype=float32)), (array([0.6201913 , 0.11433277, 0.776074  ], dtype=float32), array([0.6201913 , 0.11433277, 0.776074  ], dtype=float32))]
    # [[0.9795356522922227, 1.0938137944188668], [1.18674240320129, 0.28212079698952935], [0.07753051181414404, 0.10563547408078137], [2.1343753461250934, 0.7126653369070711], [1.6857133076881683, 1.4084879734501996], [3.111241561705335, 0.34070098638601287], [2.0831216116147973, 0.41362582214161814], [0.06487702821081996, 1.1912584037654694], [1.0053635300466017, 0.6023436764236842], [1.8482527219211597, 1.3054078608735808], [1.9760047390673172, 1.3707564441519273], [0.8593576470248048, 1.2535690353201518], [0.5831925188737834, 1.4966416349723537], [2.1598081185290696, 0.3385186676060782], [2.9762524873070286, 1.1480256166912846], [0.797781198817058, 0.3350696705101968], [1.6279755559695577, 0.04031090325563127], [0.6517864648455095, 0.6670943743580349], [1.1754896614118404, 0.72818257378463], [0.8721963041211681, 0.9217186960372356]]
    # [[0.06563908835425439, -0.7720977442273214], [1.1631756158209972, -0.5837777859293005], [1.4623161383480292, -0.6805002113446853], [0.8134014425441641, 0.10330027353143034], [-0.2344741015305116, 0.1101995609349184], [1.5890313063193384, -0.09638451972966194], [1.1295468453653965, -0.17515950699052246], [0.5575299272155383, -1.6672933392984661], [0.7310732519781463, -0.7147024645890313], [-0.05056714785634639, 0.24431383831279307], [-0.02607609891451887, 0.39609874371008263], [-0.08251030076443641, -0.921431341223975], [-0.18151691350416324, -1.2629821798036653], [1.2330845321982808, -0.2181486299694734], [1.2191132783854026, 0.8062542287883501], [1.1290701649811263, -0.7465210082205535], [1.4731870661452182, -0.5305335903393239], [0.7447836793103239, -0.9915328141102466], [0.527914704953845, -0.5846415097766168], [0.3230757516333158, -0.8764978989194223]]
    test_data = []
    m_data = []
    n_data = []
    # for i in range(20):
    #     m = np.random.rand() * np.pi
    #     n = np.random.rand() * np.pi / 2
    #     m_data.append(m)
    #     n_data.append(n)
    #     x = np.cos(m) * np.sin(n)
    #     y = np.sin(m) * np.sin(n)
    #     z = np.cos(n)
    #     test_data.append((np.array([x, y, z]), np.array([x, y, z])))

    train(min_loss=0.001, max_try_times=10)

    test()
