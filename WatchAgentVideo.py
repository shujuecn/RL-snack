import os

os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
import torch
import cv2
import numpy as np
import config.config as basic_config
from Game import GameEnvironment
from model import QNetwork, get_network_input
from pathlib import Path


def drawboard(win, snake: GameEnvironment, apple: GameEnvironment, block_size: int, windowwidth: int, windowheight: int):
    win.fill((0, 0, 0))

    # 绘制蛇身体渐变色
    for i, pos in enumerate(snake.prevpos[:-1]):
        gradient_color = (
            int(255 * (i + 1) / len(snake.prevpos)),
            int(255 * i / len(snake.prevpos)),
            0,
        )
        pygame.draw.rect(
            win,
            gradient_color,
            (pos[0] * block_size, pos[1] * block_size, block_size, block_size),
        )

    # 蛇头
    head_pos = snake.prevpos[-1]
    head_radius = int(block_size / 2)
    head_center = (
        head_pos[0] * block_size + head_radius,
        head_pos[1] * block_size + head_radius,
    )
    pygame.draw.circle(win, (0, 255, 0), head_center, head_radius)

    # 苹果
    pygame.draw.rect(
        win,
        (255, 0, 0),
        (
            apple.position[0] * block_size,
            apple.position[1] * block_size,
            block_size,
            block_size,
        ),
    )

    # 背景网格
    for x in range(0, windowwidth // 2, block_size):
        pygame.draw.line(win, (50, 50, 50), (x, 0), (x, windowheight))
    for y in range(0, windowheight, block_size):
        pygame.draw.line(win, (50, 50, 50), (0, y), (windowwidth // 2, y))

    # 返回 numpy array
    return pygame.surfarray.array3d(win)


def run_snake_game_video_info(
    model, video_path: Path | str, max_frames: int = 2000, verbose: bool = True
):
    gridsize = basic_config.GRIDSIZE
    block_size = basic_config.BLOCKSIZE
    speed = basic_config.PLAYERSPEED

    board = GameEnvironment(gridsize, nothing=0.0, dead=-10.0, apple=10.0)
    windowwidth = gridsize * block_size * 2
    windowheight = gridsize * block_size

    pygame.init()
    win = pygame.Surface((windowwidth, windowheight))  # 无窗口

    # 视频 writer，帧率用 speed
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, speed, (windowwidth, windowheight))

    # prev_len_of_snake = 0
    allRewards = 0
    frame_count = 0
    episode_count = 1

    if verbose:
        print("游戏已加载，开始 AI 玩游戏...")

    runGame = True
    while runGame and frame_count < max_frames:
        state_0 = get_network_input(board.snake, board.apple)
        state = model(state_0)
        action = int(torch.argmax(state).item())

        reward, done, len_of_snake = board.update_boardstate(action)
        allRewards += reward

        # 游戏画面
        game_frame = drawboard(
            win, board.snake, board.apple, block_size, windowwidth, windowheight
        )
        game_frame = np.transpose(game_frame, (1, 0, 2))
        game_frame = cv2.cvtColor(game_frame, cv2.COLOR_RGB2BGR)

        # 右边信息面板
        info_frame = np.zeros_like(game_frame)
        info_x = windowwidth // 2 + 10
        y = 30
        line_height = 30
        cv2.putText(
            info_frame,
            f"Episode: {episode_count}",
            (info_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        y += line_height
        cv2.putText(
            info_frame,
            f"Frame: {frame_count}",
            (info_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        y += line_height
        cv2.putText(
            info_frame,
            f"Snake Length: {len_of_snake}",
            (info_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        y += line_height
        cv2.putText(
            info_frame,
            f"Reward: {int(allRewards)}",
            (info_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        y += line_height
        cv2.putText(
            info_frame,
            f"Action: {action}",
            (info_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
        y += line_height
        status = "ALIVE" if not board.game_over else "DEAD"
        cv2.putText(
            info_frame,
            f"Status: {status}",
            (info_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        # 合并左右画面
        combined = np.hstack(
            [game_frame[:, : windowwidth // 2], info_frame[:, windowwidth // 2 :]]
        )

        out.write(combined)
        frame_count += 1

        if board.game_over:
            if verbose:
                print(f"第 {episode_count} 轮: 蛇死亡，长度 {len_of_snake}，重置游戏")
            episode_count += 1
            # prev_len_of_snake = len_of_snake
            allRewards = 0
            board.resetgame()

    out.release()
    pygame.quit()
    if verbose:
        print(
            f"游戏结束，视频已保存到 {video_path}，共 {frame_count} 帧, 共 {episode_count} 轮"
        )


if __name__ == "__main__":
    video_save_dir = Path("./video")
    video_save_dir.mkdir(parents=True, exist_ok=True)

    use_ai = True
    if use_ai:
        model = QNetwork(input_dim=10, hidden_dim=20, output_dim=5)
        model.load_state_dict(torch.load("./dir_chk/Snake_60000"))
        run_snake_game_video_info(
            model,
            video_path=video_save_dir / "snake_ai_info.mp4",
            max_frames=1000,
            verbose=True,
        )
