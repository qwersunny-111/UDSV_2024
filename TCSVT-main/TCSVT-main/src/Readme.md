1. cylinder.py
    圆柱投影

2. frontview.py
    前三固连相机拼接，frame_idx为选取的参数帧，动态O

3. rearview.py
    后五固连相机拼接，frame_idx为选取的参数帧，动态O

4. stitch_dynamic.py
    仅针对拼接过程的时域约束
    动态区域拼接，动态O，输出结果为图像序列，每帧图像的横向宽度一般不一致

5. meshflow_stitch_boost.py
    IROS所属动态双相机拼接
    固定O，输出为经过变形的左右两视频，需要根据O进行对齐与重叠区域处理

6. final.py
    将前3相机，后5相机以及动态区域堆叠为全景图，注意区分左右