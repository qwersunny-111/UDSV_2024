"""
将保存了图像帧序列的文件夹转换为 mp4 视频
输入： 图像序列文件夹
输出： 用于 HJ 项目的 mp4 视频文件
"""

import os
import subprocess

#! 定义整体参数
ROOT_DIR = "/home/lianghao/workspace/TCSVT/experiments/"   # 数据集根目录
SEQUENCES_DIR = os.path.join(ROOT_DIR, 'sequences')
VIDEO_DIR = os.path.join(ROOT_DIR, 'videos')
FRAME_RATE = 30

if __name__ == '__main__':
    os.makedirs(VIDEO_DIR, exist_ok=True)
    sequences = os.listdir(SEQUENCES_DIR)
    print(sequences)

    for s in sequences:
        print('-----------------\nStart converting sequence {} to video...'.format(s))
        image_pattern = os.path.join(SEQUENCES_DIR, s, '%06d.jpg')
        output_video = os.path.join(VIDEO_DIR, s + '.mp4')

        # 构建ffmpeg命令
        cmd = [
            'ffmpeg',                            # ffmpeg命令
            '-framerate', str(FRAME_RATE),   # 帧率
            '-i', image_pattern,            # 图片序列的命名规则
            '-c:v', 'libx264',               # 视频编码器
            '-r', str(FRAME_RATE),         # 输出视频的帧率
            output_video                        # 输出的视频文件名
        ]

        # 调用ffmpeg命令
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)   # 打印ffmpeg命令的输出
        print(result.stderr)

    print('Finish converting all sequences to videos.')