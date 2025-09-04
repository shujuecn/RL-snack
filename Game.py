# 这个文件定义了强化学习所需的一切“硬件”：智能体（蛇）、目标（苹果）和它们所在的宇宙（游戏环境）。

import numpy as np
import torch
import config.config as basic_config

# 贪吃蛇初始长度
PLAYERSIZE = basic_config.PLAYERSIZE


class SnakeClass:
    def __init__(self, gridsize: int):
        """
        初始化贪吃蛇位置、方向、轨迹和长度
        """
        self.position = np.array([gridsize // 2, gridsize // 2]).astype("float")
        self.direction = np.array([1.0, 0.0])
        self.prevpos = [np.array([gridsize // 2, gridsize // 2]).astype("float")]
        self.gridsize = gridsize
        self.len = PLAYERSIZE

    def __len__(self):
        return self.len

    def move(self):
        """
        更新贪吃蛇位置：
            更新头部位置
            记录贪吃蛇的轨迹
        """
        self.position += self.direction
        self.prevpos.append(self.position.copy())
        self.prevpos = self.prevpos[-self.len - 1 :]

    def checkdead(self, pos: np.ndarray):
        """
        判断蛇是否死亡（撞墙或撞自己），这是触发负奖励的关键事件。
        """
        if pos[0] <= -1 or pos[0] >= self.gridsize:
            return True
        elif pos[1] <= -1 or pos[1] >= self.gridsize:
            return True
        # 贪吃蛇头部是否碰撞自己
        elif list(pos) in [list(item) for item in self.prevpos[:-1]]:
            return True
        else:
            return False

    def getproximity(self):
        """
        获取蛇头周围的环境信息，判断上下左右四个方向是否会立即导致死亡。
        这部分信息是构成“状态”的重要组成部分。
        """
        L = self.position - np.array([1, 0])
        R = self.position + np.array([1, 0])
        U = self.position - np.array([0, 1])
        D = self.position + np.array([0, 1])
        # 四个可能的新位置
        possdirections = [L, R, U, D]
        # 检查四个方向移动是否会导致贪吃蛇死亡，死亡返回1，否则返回0
        proximity = [int(self.checkdead(x)) for x in possdirections]
        return proximity

    def showState(self):
        print("Snake:")
        print("Position:", self.position)
        print("Direction:", self.direction)
        print("Previous Positions:", self.prevpos)
        print("Length:", len(self))
        print("\n")


class AppleClass:
    def __init__(self, gridsize: int):
        self.position = np.random.randint(1, gridsize, 2)
        self.score = 0
        self.gridsize = gridsize

    def eaten(self):
        self.position = np.random.randint(1, self.gridsize, 2)
        self.score += 1

    def showState(self):
        print("Apple:")
        print("Position:", self.position)
        print("Score:", self.score)
        print("\n")


class GameEnvironment:
    def __init__(self, gridsize: int, nothing: float, dead: float, apple: float):
        self.snake = SnakeClass(gridsize)
        self.apple = AppleClass(gridsize)
        self.game_over = False
        self.gridsize = gridsize
        self.reward_nothing = nothing  # 每一步的微小负奖励，鼓励智能体尽快找到苹果
        self.reward_dead = dead  # 撞墙或撞自己时的负奖励
        self.reward_apple = apple  # 吃到苹果时的正奖励
        self.time_since_apple = 0  # 记录自上次吃到苹果以来的时间步数
        # 定义移动方向向量
        self.player_moves = {
            "L": np.array([-1.0, 0.0]),
            "R": np.array([1.0, 0.0]),
            "U": np.array([0.0, -1.0]),
            "D": np.array([0.0, 1.0]),
        }

    def resetgame(self):
        # 重置游戏状态
        self.snake.position = np.random.randint(1, self.gridsize, 2).astype("float")
        self.apple.position = np.random.randint(1, self.gridsize, 2).astype("float")
        self.snake.prevpos = [self.snake.position.copy().astype("float")]  # 重置轨迹
        self.apple.score = 0
        self.snake.len = PLAYERSIZE
        self.game_over = False

    def get_boardstate(self):
        return [
            self.snake.position,
            self.snake.direction,
            self.snake.prevpos,
            self.apple.position,
            self.apple.score,
            self.game_over,
        ]

    def update_boardstate(self, move: torch.Tensor | int):
        reward = self.reward_nothing
        Done = False  # 游戏是否结束的标志
        # 0:Left 1:Right 2:Up 3:Down
        # 如果方向要向左，且当前方向不为右，则更新方向为左（避免碰到自己）
        if move == 0:
            if not (self.snake.direction == self.player_moves["R"]).all():
                # 向左移动
                self.snake.direction = self.player_moves["L"]
        if move == 1:
            if not (self.snake.direction == self.player_moves["L"]).all():
                # 向右移动
                self.snake.direction = self.player_moves["R"]
        if move == 2:
            if not (self.snake.direction == self.player_moves["D"]).all():
                # 向上移动
                self.snake.direction = self.player_moves["U"]
        if move == 3:
            if not (self.snake.direction == self.player_moves["U"]).all():
                # 向下移动
                self.snake.direction = self.player_moves["D"]

        self.snake.move()
        self.time_since_apple += 1
        # 经过了100步没有吃到苹果则结束游戏，避免原地转圈
        if self.time_since_apple == 100:
            self.game_over = True
            reward = self.reward_dead
            self.time_since_apple = 0
            Done = True

        if self.snake.checkdead(self.snake.position):
            # 撞墙或撞自己
            self.game_over = True
            reward = self.reward_dead
            self.time_since_apple = 0
            Done = True
        elif (self.snake.position == self.apple.position).all():
            # 吃到苹果
            self.apple.eaten()
            self.snake.len += 1
            self.time_since_apple = 0
            reward = self.reward_apple

        len_of_snake = len(self.snake)
        return reward, Done, len_of_snake

    def showState(self):
        print("Game Environment:")
        print("Snake:")
        print("Position:", self.snake.position)
        print("Direction:", self.snake.direction)
        print("Previous Positions:", self.snake.prevpos)
        print("Length:", len(self.snake))
        print("Apple:")
        print("Position:", self.apple.position)
        print("Score:", self.apple.score)
        print("Game Over:", self.game_over)
        print("\n")


if __name__ == "__main__":
    gridsize = 10

    snake = SnakeClass(gridsize)
    snake.showState()
    snake.move()
    snake.showState()

    apple = AppleClass(gridsize)
    apple.showState()
    apple.eaten()
    apple.showState()

    env = GameEnvironment(gridsize, -0.1, -1, 1)
    env.showState()
    env.resetgame()
    env.update_boardstate(1)
    env.showState()
