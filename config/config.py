# 贪吃蛇初始长度
PLAYERSIZE = 0
# 贪吃蛇移动速度（视频帧率/模拟速度，可用在保存视频时）
PLAYERSPEED = 50
# 贪吃蛇默认移动方向
PLAYERDIRECTION = 1
# 游戏盘大小
GRIDSIZE = 15
# 格子大小
BLOCKSIZE = 20

# 经验回放池容量
MEMORYMAX = 10000         # 增大经验池，提高训练稳定性
# 学习率
MYLR = 0.0005             # 调高学习率，加快训练收敛

# 总训练轮数
NUM_EPISODES = 50000      # 适中轮数，快速看到收敛效果
# 每个episode进行学习更新的次数
NUM_UPDATES = 500
# 打印训练日志的频率，每经过PRINT_EVERY轮打印一次训练日志
PRINT_EVERY = 5
# 每轮内游戏的次数
GAMES_IN_EPISODE = 100     # 每轮积累更多经验，加快学习

# 从经验回放池中采样的batch大小
BATCH_SIZE = 256           # 增大 batch，提高梯度稳定性

# 探索率
epsilon = 1.0             # 训练初期高探索
EPSILON_FINAL = 0.05      # 训练后期保留少量探索
EPSILON_DECAY = 5000      # ε 衰减轮数

# 折扣因子
GAMMA = 0.95              # 保留短期和长期奖励权衡


