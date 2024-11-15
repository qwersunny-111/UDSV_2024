# Desc: 该文件是为了实现在未稳定情况下图像fft频率可视化,以及计算前后帧的位移等
# Date: 2024-06-24
# Author: lianghao

# 数据修改顺序：390 319 377 378

import os
import cv2
from warp import video2frame_one_seq
from warp import load_video, save_video
import torch
import sys
import numpy as np
import math
import copyreg
import matplotlib.pyplot as plt

h_size = 480
w_size = 640

def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)

copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)

# 这段代码的目的是计算一个变换矩阵 M 对图像进行变换后的宽度和高度的比例。具体来说，它通过变换图像的四个角点，然后计算变换后图像在原图像宽度和高度上的覆盖比。
def crop_metric(M):
    points = np.array([[0,0,1],[0,h_size,1], [w_size,0,1], [w_size,h_size,1]]).T
    result = np.matmul(M,points).T
    result = result[:,:2]/result[:,2:]
    w_out = 1 - max(result[0,0], result[1,0], w_size - result[2,0], w_size - result[3,0], 0)/w_size
    #这里计算的是变换后图像在原图像宽度上的覆盖比例。具体步骤是：
    # result[0,0] 和 result[1,0]：变换后左上角和左下角的 x 坐标。
    # w_size - result[2,0] 和 w_size - result[3,0]：变换后右上角和右下角的 x 坐标相对于右边界的距离。
    # 取这些值的最大值，并与 w_size 相除，得到变换后图像相对于原图像宽度的最大偏移量。
    # 用 1 减去这个值，得到变换后图像在原图像宽度上的覆盖比例。
    h_out = 1 - max(result[0,1], result[2,1], h_size - result[1,1], h_size - result[3,1], 0)/h_size
    return w_out, h_out

# https://stackoverflow.com/questions/34389125/how-to-get-the-scale-factor-of-getperspectivetransform-in-opencv
# 变换矩阵的缩放因子描述了在应用该变换后，图像或对象在不同方向上的比例变化。简而言之，缩放因子告诉我们在应用变换后，图像或对象在特定方向上被放大或缩小的倍数。本函数中，我们通过计算变换矩阵的 QR 分解来获取缩放因子。分别是x轴(R[0,0])和y轴的缩放因子。
def get_scale(M):
    h1 = M[0, 0]
    h2 = M[0, 1]
    h3 = M[0, 2]
    h4 = M[1, 0]
    h5 = M[1, 1]
    h6 = M[1, 2]
    h7 = M[2, 0]
    h8 = M[2, 1]
    QR = np.array([[h1-(h7*h3), h2-(h8*h3)], [h4-(h7*h6), h5-(h8*h6)]])
    Q, R = np.linalg.qr(QR)
    return abs(R[0,0]), abs(R[1,1])

# 根据缩放因子调整变换矩阵。这段代码的目的是对一个给定的 3x3 变换矩阵 M 进行重新缩放。具体来说，通过给定的缩放因子 sx 和 sy，调整矩阵 M 以反映新的缩放比例。
# https://stackoverflow.com/questions/21019338/how-to-change-the-homography-with-the-scale-of-the-image
def get_rescale_matrix(M, sx, sy):
    S = np.eye(3, dtype = float)
    S[0,0] = sx
    S[1,1] = sy

    S1 = np.eye(3, dtype = float)
    S1[0,0] = 1/sx
    S1[1,1] = 1/sy
    return np.matmul(M, S1)

def crop_rm_outlier(crop):
    crop = np.array(crop)
    crop = crop[crop >= 0.5]
    return sorted(crop)[5:]

def crop_video(in_path, out_path, crop_ratio):
    frame_array, fps, size = load_video(in_path)
    hs = int((1-crop_ratio)*1080) + 1
    he = int(crop_ratio*1080) - 1
    ws = int((1-crop_ratio)*1920) + 1
    we = int(crop_ratio*1920) - 1
    for i in range(len(frame_array)):
        frame_array[i] = cv2.resize(frame_array[i][hs:he,ws:we,:], size, interpolation = cv2.INTER_LINEAR)
    save_video(out_path, frame_array, fps, size= size)


def plot_fft(signal, title, save_path_prefix):
    """绘制信号的 FFT 能量谱"""
    fft = np.fft.fft(signal)
    fft = abs(fft) ** 2
    fft = np.delete(fft, 0)
    fft = fft[:len(fft) // 2]
    # 视频的帧率是30FPS，因此我们可以使用 30 作为 FFT 的采样频率。
    freqs = np.fft.fftfreq(len(signal), d=30.0)[:len(fft)]
    file_path =f'{save_path_prefix}_{title.replace(" ", "_").lower()}_power.png'
    plt.figure()
    plt.plot(freqs, fft)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid()
    plt.savefig(file_path)
    # plt.show()

def is_frame_file(filename):
    # 检查文件扩展名是否是目标帧文件类型
    # return filename.lower().endswith(('.jpg', '.jpeg', '.png'))
    return filename.lower().endswith(('.png'))

def metrics(in_src, out_src, package, crop_scale = False, re_compute = False):
    load_dic = None
    if re_compute and os.path.exists(package):
        print("Start load")
        load_dic = torch.load(package)
        print("Finish load")
    dic = {
        'M': None,
        'CR_seq': [],
        'DV_seq': [],
        'SS_t': None,
        'SS_r': None,
        'w_crop':[],
        'h_crop':[],
        'distortion': [],
        'count': 0,
        'in_sift': {},
        'out_sift': {},
        'fft_t': {},
        'fft_r': {}
        }

    if load_dic is not None:
        dic["in_sift"] = load_dic["in_sift"]
        dic["out_sift"] = load_dic["out_sift"]

    # 获取并过滤输入源目录中的帧文件
    frameList= sorted([f for f in os.listdir(in_src) if is_frame_file(f)])

    # Create brute-force matcher object
    bf = cv2.BFMatcher()

    # Apply the homography transformation if we have enough good matches 
    MIN_MATCH_COUNT = 10 #10

    ratio = 0.7 #0.7
    thresh = 5.0 #5.0

    Pt = np.asarray([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    P_seq = []
    count = 1
    for index, f in enumerate(frameList, 0):
        if f.endswith('.png'):
            if count == len(frameList):
                break
            sift = cv2.SIFT_create()  
            # Load the images in gray scale
            img1 = cv2.imread(os.path.join(in_src, f), 0)  
            img1 = cv2.resize(img1, (w_size,h_size), interpolation = cv2.INTER_LINEAR)

            f_path = f[:-9] + '%05d.png' % (int(f[-9:-4])+1)
            if f_path in dic["out_sift"]:
                keyPoints1o, descriptors1o = dic["out_sift"][f_path]  
            else:
                img1o = cv2.imread(os.path.join(out_src, f_path), 0)
                img1o = cv2.resize(img1o, (w_size,h_size), interpolation = cv2.INTER_LINEAR)
                keyPoints1o, descriptors1o = sift.detectAndCompute(img1o, None)
                dic["out_sift"][f_path] = (keyPoints1o, descriptors1o)          
            
            sift = cv2.SIFT_create()   
            
            if f in dic["in_sift"]:
                keyPoints1, descriptors1 = dic["in_sift"][f]
            else:
                # Detect the SIFT key points and compute the descriptors for the two images
                keyPoints1, descriptors1 = sift.detectAndCompute(img1, None)
                dic["in_sift"][f] = (keyPoints1, descriptors1)

            # Match the descriptors
            matches = bf.knnMatch(descriptors1, descriptors1o, k=2)

            # Select the good matches using the ratio test
            goodMatches = []

            for m, n in matches:
                if m.distance < ratio * n.distance:
                    goodMatches.append(m)

            if len(goodMatches) > MIN_MATCH_COUNT:
                # Get the good key points positions
                sourcePoints = np.float32([ keyPoints1[m.queryIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
                destinationPoints = np.float32([ keyPoints1o[m.trainIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
                
                M, mask = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=thresh)
                im_dst = cv2.warpPerspective(img1, M, (w_size,h_size))  

                cm = []
                for i in range(6):
                    for j in range(6):
                        hs = int(h_size * (0.2 + 0.1 * i))
                        he = int(h_size * (0.3 + 0.1 * i))
                        ws = int(w_size * (0.2 + 0.1 * j))
                        we = int(w_size * (0.3 + 0.1 * j))
                        cm.append(np.corrcoef(img1o[hs:he, ws:we].flat, im_dst[hs:he, ws:we].flat))
                dic["distortion"].append(cm)

                if crop_scale:
                    sx, sy = get_scale(M)
                    M_scale = get_rescale_matrix(M, sx, sy)
                    w_crop, h_crop = crop_metric(M_scale)
                else:
                    w_crop, h_crop = crop_metric(M)
                dic["w_crop"].append(w_crop)
                dic["h_crop"].append(h_crop)

            # Obtain Scale, Translation, Rotation, Distortion value
            # Obtain Scale, Translation, Rotation, Distortion value
            #这种计算缩放因子的方法不如QR分解准确，因为它只是简单地计算了变换矩阵的对角元素。QR分解是先将变换矩阵分解为正交矩阵和上三角矩阵，然后再计算对角元素。后面DV也是相同的问题，其计算特征值时
            #单应性矩阵（Homography Matrix）通常是一个 3x3 矩阵，用于描述平面之间的变换关系。单应性矩阵的最后一行通常不是 [0, 0, 1]，这会影响我们直接从 2x2 子矩阵中提取特征值并计算缩放因子和畸变率的准确性。
            #解决方法：todo 目前已知QR分解可以计算缩放因子
            sx = M[0, 0]
            sy = M[1, 1]
            scaleRecovered = math.sqrt(np.abs(sx*sy))

            w, _ = np.linalg.eig(M[0:2,0:2])
            w = np.sort(w)[::-1]
            DV = w[1]/w[0]
            #pdb.set_trace()

            dic["CR_seq"].append(1.0/scaleRecovered)
            dic["DV_seq"].append(DV)  

            # For Stability score calculation
            if count < len(frameList):
                
                matches = bf.knnMatch(descriptors1, descriptors1o, k=2)
                goodMatches = []

                for m, n in matches:
                    if m.distance < ratio * n.distance:
                        goodMatches.append(m)

                if len(goodMatches) > MIN_MATCH_COUNT:
                    # Get the good key points positions
                    sourcePoints = np.float32([ keyPoints1[m.queryIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
                    destinationPoints = np.float32([ keyPoints1o[m.trainIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
                    
                    # Obtain the homography matrix
                    M, mask = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=thresh)

                P_seq.append(np.matmul(Pt, M))
                Pt = np.matmul(Pt, M)
            if count % 10 ==0:
                sys.stdout.write('\rFrame: ' + str(count) + '/' + str(len(frameList)))
                sys.stdout.flush()
            dic["count"] = count
            count += 1

    # Make 1D temporal signals
    P_seq_t = np.asarray([1])
    P_seq_r = np.asarray([1])

    #pdb.set_trace()
    #最新DIFRINT代码中的计算方法，已更改下面已过时，同样的问题不一定是仿射矩阵。尽管可以从单应性矩阵中提取某些元素来近似计算平移和旋转，但这种方法有其局限性。对于更准确的变换分解，通常需要使用专门的算法来处理单应性矩阵的分解。
    #可以使用奇异值分解（SVD）来分解单应矩阵，但这通常需要更复杂的数学处理。为了简单起见，可以使用 OpenCV 提供的函数 decomposeHomographyMat 来进行分解，但是需要相机内参。
    #pdb.set_trace()
    for Mp in P_seq:
        # print(Mp)
        sx = Mp[0, 0]
        sy = Mp[1, 1]
        c = Mp[0, 2]
        f = Mp[1, 2]

        transRecovered = math.sqrt(c*c + f*f)
        thetaRecovered = math.atan2(sx, sy) * 180 / math.pi

        P_seq_t = np.concatenate((P_seq_t, [transRecovered]), axis=0)
        P_seq_r = np.concatenate((P_seq_r, [thetaRecovered]), axis=0)

    P_seq_t = np.delete(P_seq_t, 0)
    P_seq_r = np.delete(P_seq_r, 0)

    # FFT
    # FFT对平移和旋转信号进行快速傅里叶变换（FFT），得到频域表示。取傅里叶变换结果的绝对值的平方，得到功率谱密度。
    fft_t = np.fft.fft(P_seq_t)
    fft_r = np.fft.fft(P_seq_r)
    # 打印结果
    # print("fft_t:", fft_t)
    # print("fft_r:", fft_r)

    # 计算频率
    # 视频的帧率是30FPS，因此我们可以使用 30 作为 FFT 的采样频率。
    freq_t = np.fft.fftfreq(len(P_seq_t),30)
    freq_r = np.fft.fftfreq(len(P_seq_r),30)

    # 删除直流分量，取一半的频谱，因为是对称的
    fft_t = np.delete(fft_t, 0)
    fft_r = np.delete(fft_r, 0)
    freq_t = np.delete(freq_t, 0)
    freq_r = np.delete(freq_r, 0)
    fft_t = fft_t[:int(len(fft_t)/2)]
    fft_r = fft_r[:int(len(fft_r)/2)]
    freq_t = freq_t[:int(len(freq_t)/2)]
    freq_r = freq_r[:int(len(freq_r)/2)] 

    # 绘制频谱
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.stem(freq_t, np.abs(fft_t), 'b', markerfmt=" ", basefmt="-b")
    plt.title('Translation FFT Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.stem(freq_r, np.abs(fft_r), 'r', markerfmt=" ", basefmt="-r")
    plt.title('Rotation FFT Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    #注意需要修改保存路径
    plt.savefig('/home/lianghao/workspace/TCSVT/experiments/results/fft/case5_fft.png')
    # plt.show()

    # 计算功率谱
    fft_t = abs(fft_t)**2
    # print(fft_t)
    fft_r = abs(fft_r)**2

    dic["fft_t"] = fft_t
    dic["fft_r"] = fft_r
    #计算频谱前 5 个频率分量的能量占总能量的比例。这可以用来分析信号的低频成分在总能量中的占比，通常用于评估信号的平稳性或抖动。
    SS_t = np.sum(fft_t[:5])/np.sum(fft_t)  
    SS_r = np.sum(fft_r[:5])/np.sum(fft_r)

    dic["CR_seq"] = np.array(dic["CR_seq"])
    dic["DV_seq"] = np.array(dic["DV_seq"])
    dic["w_crop"] = np.array(dic["w_crop"])
    dic["h_crop"] = np.array(dic["h_crop"])
    dic["distortion"] = np.array(dic["distortion"])
    dic["SS_t"] = SS_t
    dic["SS_r"] = SS_r
    
    if not (re_compute and os.path.exists(package)):
        torch.save(dic, package)

    #计算 dic["DV_seq"] 的绝对值。
    DV_seq = np.absolute(dic["DV_seq"])
    #过滤 DV_seq 中的值，只保留在 0.5 和 1 之间的值。
    DV_seq = DV_seq[np.where((DV_seq >= 0.5) & (DV_seq <= 1))]
    #计算 DV_seq 中的最小值，忽略 NaN 值。
    Distortion = str.format('{0:.4f}', np.nanmin(DV_seq))
    #计算 DV_seq 的平均值，忽略 NaN 值。
    Distortion_avg = str.format('{0:.4f}', np.nanmean(DV_seq))

    Trans = str.format('{0:.4f}', dic["SS_t"])
    Rot = str.format('{0:.4f}', dic["SS_r"])

    w_crop = crop_rm_outlier(dic["w_crop"])
    h_crop = crop_rm_outlier(dic["h_crop"])

    FOV = str.format( '{0:.4f}', min(np.nanmin(w_crop), np.nanmin(h_crop)) )
    FOV_avg = str.format( '{0:.4f}', (np.nanmean(w_crop)+np.nanmean(h_crop)) / 2 )

    Correlation_avg = str.format( '{0:.4f}', np.nanmean(dic["distortion"][10:]) )
    Correlation_min = str.format( '{0:.4f}', np.nanmin(dic["distortion"][10:]) )

    # Print results
    print('\n***Distortion value (Avg, Min):')
    print(Distortion_avg +' | '+  Distortion)
    print('***Stability Score (Avg, Trans, Rot):')
    print(str.format('{0:.4f}',  (dic["SS_t"]+dic["SS_r"])/2) +' | '+ Trans +' | '+ Rot )
    print("=================")
    print('***FOV ratio (Avg, Min):')
    print( FOV_avg +' | '+ FOV )
    print('***Correlation value (Avg, Min):')
    print( Correlation_avg +' | '+ Correlation_min , "\n")  

    # 绘制 FFT 功率谱图
    plot_fft(P_seq_t, "Translation Power Spectrum",'/home/lianghao/workspace/TCSVT/experiments/results/fft/case5')
    plot_fft(P_seq_r, "Rotation Power Spectrum",'/home/lianghao/workspace/TCSVT/experiments/results/fft/case5')

    dic['in_sift'] = 0
    dic['out_sift'] = 0
    torch.save(dic, package[:-3]+"_light.pt") 
    return float(FOV)

if __name__ == "__main__":
    data_path = os.path.join("/home/lianghao/workspace/TCSVT/experiments/data/")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    in_video = "/home/lianghao/workspace/TCSVT/experiments/data/videos/case5.mp4"
    #需要把video的名字提取出来，不然是整个路径
    video_name_with_ext = os.path.basename(in_video)
    video_name = os.path.splitext(video_name_with_ext)[0]
    # print(video_name)
    in_folder = os.path.join(data_path, "sequences",video_name)
    if not os.path.exists(in_folder):
        os.makedirs(in_folder)
    print("Convert video to frames")
    video2frame_one_seq(in_video, in_folder)

    package = os.path.join(data_path, "fft.pt")
    FOV = metrics(in_folder, in_folder, package)