import torch
import torch.nn as nn
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
import config.config as basic_config
from model import QNetwork, get_network_input
from Game import GameEnvironment
from collections import deque
from replay_buffer import ReplayMemory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_episode(
    num_games, board: GameEnvironment, model: QNetwork, memory: ReplayMemory
):
    """
    模拟一局游戏，返回总奖励、平均蛇长度和最大蛇长度
    1. 根据当前状态选择动作（ε-贪婪策略）
    2. 执行动作，观察奖励和下一个状态
    3. 将经验存储到经验回放缓冲区
    4. 如果游戏结束，重置游戏环境
    5. 重复上述步骤直到玩完指定局数
    6. 计算并返回总奖励、平均蛇长度和最大蛇长度
    7. 在训练过程中，逐渐减少ε值以减少随机动作的选择
    8. 使用经验回放缓冲区中的样本进行模型训练
    """
    run = True
    games_played = 0  # 已玩的游戏局数
    total_reward = 0  # 总奖励
    episode_games = 0  # 当前回合玩的游戏局数
    len_array = []  # 存储每局游戏的蛇长度

    while run:
        # 当前状态
        state = get_network_input(board.snake, board.apple)
        state = state.to(device)
        # 根据ε-贪婪策略选择动作
        action_0 = model(state)
        rand = np.random.uniform(0, 1)
        if rand > basic_config.epsilon:
            # 让Q网络预测当前状态下所有动作的Q值，然后选择Q值最大的那个动作
            action = torch.argmax(action_0)
        else:
            # 随机选一个动作，这有助于发现新的、可能更好的玩法
            action = np.random.randint(0, 5)

        # 执行动作，观察奖励和下一个状态
        reward, done, len_of_snake = board.update_boardstate(action)
        # 下一个状态
        next_state = get_network_input(board.snake, board.apple)
        next_state = next_state.to(device)
        # 将经验存储到经验回放缓冲区
        memory.push(state, action, reward, next_state, done)

        total_reward += reward

        episode_games += 1

        if board.game_over:
            games_played += 1
            len_array.append(len_of_snake)
            board.resetgame()

            if num_games == games_played:
                run = False

    avg_len_of_snake = np.mean(len_array)
    max_len_of_snake = np.max(len_array)
    return total_reward, avg_len_of_snake, max_len_of_snake


def learn(
    memory, model: QNetwork, optimizer: torch.optim.Optimizer, criterion: nn.MSELoss
):
    """
    从经验回放缓冲区中采样并更新模型
    """
    total_loss = 0

    # 更新模型 NUM_UPDATES 次
    for _ in range(basic_config.NUM_UPDATES):
        optimizer.zero_grad()
        sample = memory.sample(basic_config.BATCH_SIZE)  # 采样一个批次的经验

        states, actions, rewards, next_states, dones = sample
        states = torch.cat([x.unsqueeze(0) for x in states], dim=0)
        states = states.to(device)
        actions = torch.LongTensor(actions)
        actions = actions.to(device)
        rewards = torch.FloatTensor(rewards)
        rewards = rewards.to(device)
        next_states = torch.cat([x.unsqueeze(0) for x in next_states])
        next_states = next_states.to(device)
        dones = torch.FloatTensor(dones)
        dones = dones.to(device)

        # 前向传播：计算当前状态下采取动作的Q值
        q_local = model.forward(states)
        # 前向传播：计算下一个状态的Q值
        next_q_value = model.forward(next_states)

        # 预测Q值 (Q_expected)：对于采样出的每一个 state，用Q网络预测它对应的 action 的Q值。
        Q_expected = (
            q_local.gather(1, actions.unsqueeze(0).transpose(0, 1))
            .transpose(0, 1)
            .squeeze(0)
        )

        # 计算下一个状态的最大Q值 (max_a' Q(next_state, a'))，并考虑游戏是否结束
        # 如果游戏结束，max_a' Q(next_state, a') = 0
        Q_targets_next = torch.max(next_q_value, 1)[0] * (
            torch.ones(dones.size(), device=device) - dones
        )

        # 目标Q值（Q_targets）：计算目标Q值（希望网络能预测出的“正确答案”）
        # Reward + γ * max(Q(next_state, all_actions))
        # GAMMA 是折扣因子（<1），表示未来奖励的重要性不如当前奖励
        Q_targets = rewards + basic_config.GAMMA * Q_targets_next

        loss = criterion(Q_expected, Q_targets)

        total_loss += loss
        loss.backward()
        optimizer.step()

    return total_loss


def train(
    model: QNetwork,
    board: GameEnvironment,
    memory: ReplayMemory,
    optimizer: torch.optim.Optimizer,
    MSE: torch.nn.MSELoss,
):
    """
    训练深度Q网络模型
    """
    print("Training started on {}".format(device))

    # 固定长度队列：只保留最近100个回合的总得分，用于计算滑动平均分
    scores_deque = deque(maxlen=100)
    scores_array = []  # 记录每个回合的总得分
    avg_scores_array = []  # 记录每个回合的滑动平均分

    avg_len_array = []  # 记录每个回合中蛇的平均长度
    avg_max_len_array = []  # 记录每个回合中蛇达到的最大长度

    time_start = time.time()
    temp_avg_len = 0  # 用于判断模型是否有进步

    # 开始训练循环，共进行 NUM_EPISODES 个回合
    for i_episode in range(basic_config.NUM_EPISODES + 1):
        # 每个回合(Episode)包含两个阶段:
        # 1. 数据收集：玩 basic_config.GAMES_IN_EPISODE 局游戏，并将经验存入记忆库
        total_reward, avg_len, max_len = run_episode(
            basic_config.GAMES_IN_EPISODE, board, model, memory
        )

        # 记录当前回合的统计数据
        scores_deque.append(total_reward)
        scores_array.append(total_reward)
        avg_len_array.append(avg_len)
        avg_max_len_array.append(max_len)

        # 计算最近100个回合的平均得分
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        # 2. 模型学习：从记忆库中采样数据，更新一次网络权重
        total_loss = learn(memory, model, optimizer, MSE)

        dt = int(time.time() - time_start)

        # 每隔 PRINT_EVERY 个回合，打印一次训练日志
        if i_episode % basic_config.PRINT_EVERY == 0 and i_episode > 0:
            print(
                "回合: {:6}, 损失: {:.4f}, 本回合总奖励 (玩{}局): {:.2f}, "
                "平均长度/{}局: {:.2f}, 最大长度/{}局: {:.2f}  已训练时长: "
                "{:02}:{:02}:{:02}".format(
                    i_episode,
                    total_loss,
                    basic_config.GAMES_IN_EPISODE,
                    total_reward,
                    basic_config.GAMES_IN_EPISODE,
                    avg_len,
                    basic_config.GAMES_IN_EPISODE,
                    max_len,
                    dt // 3600,
                    dt % 3600 // 60,
                    dt % 60,
                )
            )

        # 截断经验回放池，防止其无限增大
        memory.truncate()

        # 保存模型：当平均蛇长度有提升时，保存当前模型权重
        if i_episode > 0 and avg_len > temp_avg_len:
            torch.save(model.state_dict(), "dir_chk/Snake_{}".format(i_episode))
            temp_avg_len = avg_len  # 更新最佳平均长度

    return scores_array, avg_scores_array, avg_len_array, avg_max_len_array


def plot_scores(scores, avg_scores, avg_len_of_snake, max_len_of_snake):
    # 绘制得分和平均得分的折线图
    plt.figure()
    plt.plot(np.arange(1, len(scores) + 1), scores, label="Reward")
    plt.plot(np.arange(1, len(avg_scores) + 1), avg_scores, label="Avg Reward")
    plt.legend()
    plt.ylabel("Reward")
    plt.xlabel("Episodes #")
    plt.show()

    # 绘制平均蛇长度和最大蛇长度的折线图
    plt.figure()
    plt.plot(
        np.arange(1, len(avg_len_of_snake) + 1),
        avg_len_of_snake,
        label="Avg Len of Snake",
    )
    plt.plot(
        np.arange(1, len(max_len_of_snake) + 1),
        max_len_of_snake,
        label="Max Len of Snake",
    )
    plt.legend()
    plt.ylabel("Length of Snake")
    plt.xlabel("Episodes #")
    plt.show()

    # 绘制最大蛇长度的直方图
    plt.figure()
    sns.histplot(max_len_of_snake, bins=45, kde=True, color="green")
    plt.xlabel("Max Lengths")
    plt.ylabel("Probability")
    plt.title("Histogram of Max Lengths")
    plt.grid(True)
    plt.show()


def drawScores(scores, sample_interval):
    # 绘制得分和平均得分的折线图
    plt.figure(figsize=(20, 10))
    plt.plot(
        np.arange(1, len(scores) + 1, sample_interval),
        scores[::sample_interval],
        label="Reward",
    )
    plt.plot(
        np.arange(1, len(avg_scores) + 1, sample_interval),
        avg_scores[::sample_interval],
        label="Avg Reward",
    )
    plt.legend()
    plt.ylabel("Reward")
    plt.xlabel("Episodes")
    plt.savefig("./outputImage/drawScores.png")


def drawAvgAndMaxLen(avg_len_of_snake, max_len_of_snake, sample_interval):
    # 绘制平均蛇长度和最大蛇长度的折线图
    plt.figure(figsize=(20, 10))
    plt.plot(
        np.arange(1, len(avg_len_of_snake) + 1, sample_interval),
        avg_len_of_snake[::sample_interval],
        label="Avg Len of Snake",
    )
    plt.plot(
        np.arange(1, len(max_len_of_snake) + 1, sample_interval),
        max_len_of_snake[::sample_interval],
        label="Max Len of Snake",
    )
    plt.legend()
    plt.ylabel("Length of Snake")
    plt.xlabel("Episodes")
    plt.savefig("./outputImage/drawAvgAndMaxLen.png")


def drawMaxHist(max_len_of_snake):
    # 绘制最大蛇长度的直方图
    plt.figure(figsize=(80, 30))
    sns.histplot(max_len_of_snake, bins=45, kde=True, color="green")
    plt.xlabel("Max Lengths")
    plt.ylabel("Probability")
    plt.title("Histogram of Max Lengths")
    plt.grid(True)
    plt.savefig("./outputImage/drawMaxHist.png")


if __name__ == "__main__":
    model = QNetwork(input_dim=10, hidden_dim=20, output_dim=5).to(device)
    board = GameEnvironment(basic_config.GRIDSIZE, nothing=0, dead=-1, apple=1)
    memory = ReplayMemory(basic_config.MEMORYMAX)
    optimizer = torch.optim.Adam(model.parameters(), lr=basic_config.MYLR)
    MSE = nn.MSELoss()

    scores, avg_scores, avg_len_of_snake, max_len_of_snake = train(
        model, board, memory, optimizer, MSE
    )

    plot_scores(scores, avg_scores, avg_len_of_snake, max_len_of_snake)

    # 将scores, avg_scores, avg_len_of_snake, max_len_of_snake保存到文件中
    np.save("./npy/scores.npy", scores)
    np.save("./npy/avg_scores.npy", avg_scores)
    np.save("./npy/avg_len_of_snake.npy", avg_len_of_snake)
    np.save("./npy/max_len_of_snake.npy", max_len_of_snake)

    # 读取文件中的scores, avg_scores, avg_len_of_snake, max_len_of_snake，并绘制折线图
    scores = np.load("./npy/scores.npy")
    avg_scores = np.load("./npy/avg_scores.npy")
    avg_len_of_snake = np.load("./npy/avg_len_of_snake.npy")
    max_len_of_snake = np.load("./npy/max_len_of_snake.npy")
    drawScores(scores, 500)
    drawAvgAndMaxLen(avg_len_of_snake, max_len_of_snake, 500)
    drawMaxHist(max_len_of_snake)
