import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
# DQN reinforcement Learning

BATCH_SIZE = 32
LR = 0.001
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')  # 立杆子游戏
env = env.unwrapped
N_ACTIONS = env.action_space.n  # next possible actions
N_STATES = env.observation_space.shape[0]  # the num of states
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(),
                              int) else env.action_space.sample().shape  # to confirm the shape


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        return self.out(F.relu(self.fc1(x)))


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # the number of update's times
        self.memory_counter = 0  # a counter of memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initial momery
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:
            # choose the best action
            actions_values = self.eval_net.forward(x)
            action = torch.max(actions_values, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:
            # choose random action
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # if momery is over ,then overlap the oldest memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # update the arguments of target net
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            # update target net
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # get a batch of data from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # use action b_a to choose the value of q_eval
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape(batch,1)
        q_next = self.target_net(b_s_).detach()  # stop q_next feedbacking
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # update eval_net()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# train
dqn = DQN()
for i_episode in range(400):
    state = env.reset()
    while True:
        env.render()
        action = dqn.choose_action(state)
        # do some action ,get result
        s_, r, done, info = env.step(action)

        # edit reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(state, action, r, s_)
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()  # start learning
        if done:
            break

        state = s_
env.close()