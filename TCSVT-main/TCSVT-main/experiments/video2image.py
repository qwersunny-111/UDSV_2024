"""
将保存了视频的文件夹转换为图像序列
输入： 视频文件夹
输出： 图像文件
"""

import os
import subprocess

#! 定义整体参数
ROOT_DIR = "/home/lianghao/workspace/TCSVT/experiments/results/"   # 数据集根目录
SEQUENCES_DIR = os.path.join(ROOT_DIR, 'sequences')
VIDEO_DIR = os.path.join(ROOT_DIR, 'videos')
FRAME_RATE = 30

videos = os.listdir(VIDEO_DIR)

for v in videos:
    print('-----------------\nStart converting video {} to image sequence...'.format(v))
    input_video = os.path.join(VIDEO_DIR, v)
    output_pattern = os.path.join(SEQUENCES_DIR, v.split('.')[0], '%06d.jpg')
    os.makedirs(os.path.dirname(output_pattern), exist_ok=True)

    # 构建ffmpeg命令
    cmd = [
    'ffmpeg',                            # ffmpeg命令
    '-i', input_video,                   # 输入的视频文件名
    '-start_number', '0',                # 从0开始编号
    output_pattern                       # 输出的图像序列命名规则
    ]


    # 调用ffmpeg命令
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)   # 打印ffmpeg命令的输出
    print(result.stderr)

print('Finish converting all videos to image sequences.')