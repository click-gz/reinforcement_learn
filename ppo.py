import numpy as np
import torch
from torch import nn
import torch.nn.functional as nnf
import random
from IPython import display
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#actor
class PolicyNet(nn.Module):
    def __init__(self, state_size, hiddens, action_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hiddens)
        self.fc2 = nn.Linear(hiddens, action_size)

    def forward(self, x):
        x = nnf.softmax(self.fc2(nnf.relu(self.fc1(x))))
        return x

##critic
class ValueNet(nn.Module):
    def __init__(self, state_size, hiddens):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hiddens)
        self.fc2 = nn.Linear(hiddens, 1)

    def forward(self, x):
        x = self.fc1(x)  # [b,state_size]-->[b,hiddens]
        x = nnf.relu(x)
        x = self.fc2(x)  # [b,hiddens]-->[b,1]  评价当前的状态价值state_value
        return x

##ppo module
class PPO:
    def __init__(self, state_size, hiddens, action_size, actor_lr, critic_lr, lamb, epochs, eps, gamma):
        self.actor = PolicyNet(state_size, hiddens, action_size).to(device)
        self.critic = ValueNet(state_size, hiddens).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma  # 折扣因子
        self.lmbda = lamb  # GAE优势函数的缩放系数
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数

    def take_action(self, state):
        # 维度变换 [n_state]-->tensor[1,n_states]
        state = torch.tensor(state[np.newaxis, :]).to(device)
        # 当前状态下，每个动作的概率分布 [1,n_states]
        probs = self.actor(state)
        # 创建以probs为标准的概率分布
        action_list = torch.distributions.Categorical(probs)
        # 依据其概率随机挑选一个动作
        action = action_list.sample().item()
        return action

        # 训练
    def learn(self, transition_dict):
        # 提取数据集
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(device)
        actions = torch.tensor(transition_dict['actions']).to(device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(device).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(device).view(-1, 1)

        # 目标，下一个状态的state_value  [b,1]
        next_q_target = self.critic(next_states)
        # 目标，当前状态的state_value  [b,1]
        td_target = rewards + self.gamma * next_q_target * (1 - dones)
        # 预测，当前状态的state_value  [b,1]
        td_value = self.critic(states)
        # 目标值和预测值state_value之差  [b,1]
        td_delta = td_target - td_value

        # 时序差分值 tensor-->numpy  [b,1]
        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0  # 优势函数初始化
        advantage_list = []

        # 计算优势函数
        for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
            # 优势函数GAE的公式
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        # 正序
        advantage_list.reverse()
        # numpy --> tensor [b,1]
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(device)

        # 策略网络给出每个动作的概率，根据action得到当前时刻下该动作的概率
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # 一组数据训练 epochs 轮
        for _ in range(self.epochs):
            # 每一轮更新一次策略网络预测的状态
            log_probs = torch.log(self.actor(states).gather(1, actions))
            # 新旧策略之间的比例
            ratio = torch.exp(log_probs - old_log_probs)
            # 近端策略优化裁剪目标函数公式的左侧项
            surr1 = ratio * advantage
            # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # 策略网络的损失函数
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            # 价值网络的损失函数，当前时刻的state_value - 下一时刻的state_value
            critic_loss = torch.mean(nnf.mse_loss(self.critic(states), td_target.detach()))

            # 梯度清0
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # 反向传播
            actor_loss.backward()
            critic_loss.backward()
            # 梯度更新
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def test(self, env, play):
        #初始化游戏
        state = env.reset()[0]
        # state = torch.tensor(state[np.newaxis, :]).to(device)
        # print(state.shape)
        #记录反馈值的和,这个值越大越好
        reward_sum = 0

        #玩到游戏结束为止
        over = False
        while not over:
            #根据当前状态得到一个动作
            action = self.take_action(state)

            #执行动作,得到反馈
            state, reward, over, _, _ = env.step(action)
            reward_sum += reward

            #打印动画
            if play and random.random() < 0.2:  #跳帧
                display.clear_output(wait=True)
                plt.imshow(env.render())
                plt.show()

        return reward_sum

import gym

# ----------------------------------------- #
# 参数设置
# ----------------------------------------- #


num_episodes = 100  # 总迭代次数
gamma = 0.9  # 折扣因子
actor_lr = 1e-3  # 策略网络的学习率
critic_lr = 1e-2  # 价值网络的学习率
n_hiddens = 16  # 隐含层神经元个数
env_name = 'CartPole-v1'
return_list = []  # 保存每个回合的return

#%%
# ----------------------------------------- #
# 环境设置
# ----------------------------------------- #
env = gym.make(env_name, render_mode='rgb_array')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
#%%
# ----------------------------------------- #
# 模型构建
# ----------------------------------------- #

agent = PPO(state_size=state_size,  # 状态数
            hiddens=n_hiddens,  # 隐含层数
            action_size=action_size,  # 动作数
            actor_lr=actor_lr,  # 策略网络学习率
            critic_lr=critic_lr,  # 价值网络学习率
            lamb = 0.95,  # 优势函数的缩放因子
            epochs = 10,  # 一组序列训练的轮次
            eps = 0.2,  # PPO中截断范围的参数
            gamma=gamma,  # 折扣因子
            )
#%%
# ----------------------------------------- #
# 训练--回合更新 on_policy
# ----------------------------------------- #

for i in range(num_episodes):

    state = env.reset()[0]  # 环境重置
    done = False  # 任务完成的标记
    episode_return = 0  # 累计每回合的reward

    # 构造数据集，保存每个回合的状态数据
    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': [],
    }

    while not done:
        action = agent.take_action(state)  # 动作选择
        next_state, reward, done, _, _  = env.step(action)  # 环境更新
        # 保存每个时刻的状态\动作\...
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        # 更新状态
        state = next_state
        # 累计回合奖励
        episode_return += reward

    # 保存每个回合的return
    return_list.append(episode_return)
    # 模型训练
    agent.learn(transition_dict)

    # 打印回合信息
    print(f'iter:{i}, return:{np.mean(return_list[-10:])}')
#%%
agent.test(env, True)