import random
import numpy as np
import torch


class ReplayMemory(object):
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = []

    def push(
        self,
        state: np.ndarray | list[float] | torch.Tensor | int,
        action: torch.Tensor | int,
        reward: float,
        next_state: np.ndarray | list[float] | torch.Tensor | int,
        done: bool,
    ):
        # 用于存储经验元组 (状态, 动作, 奖励, 下一个状态, 是否结束)
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        # 从缓冲区中随机采样一个批次的经验
        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)  # 状态
            action_batch.append(action)  # 动作
            reward_batch.append(reward)  # 奖励
            next_state_batch.append(next_state)  # 下一个状态
            done_batch.append(done)  # 是否结束

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def truncate(self):
        # 用于截断缓冲区，保持其最大容量
        self.buffer = self.buffer[-self.max_size :]

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    # 经验回放缓冲区的最大容量
    max_size = 1000
    replay_memory = ReplayMemory(max_size)

    # 添加一些经验元组到缓冲区
    for i in range(10):
        state = [random.random() for _ in range(4)]  # 用随机数代替状态
        action = random.randint(0, 3)  # 随机选择一个动作
        reward = random.random()  # 随机生成奖励
        next_state = [random.random() for _ in range(4)]  # 用随机数代替下一个状态
        done = random.choice([True, False])  # 随机生成是否结束

        replay_memory.push(state, action, reward, next_state, done)

    # 输出缓冲区中的经验个数
    print("Number of experiences in replay memory:", len(replay_memory))

    # 从缓冲区中采样一个批次的经验
    batch_size = 5
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = (
        replay_memory.sample(batch_size)
    )

    # 输出采样到的经验
    print("Sampled state batch:", state_batch)
    print("Sampled action batch:", action_batch)
    print("Sampled reward batch:", reward_batch)
    print("Sampled next state batch:", next_state_batch)
    print("Sampled done batch:", done_batch)
