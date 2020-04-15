# 最簡單的詮釋: 找到 target 中的值
import numpy as np
target = np.array([1 , 2 , 3])

def f(w):
    return -np.sum((w - target) ** 2)

n_scout = 50      # 斥候人數
sigma = 0.1       # noise的標準差
alpha = 0.001     # 學習率
in_ = np.random.randn(3) # 對target的初始猜測

for iteration in range(0 , 300):
    noise = np.random.randn(n_scout , 3)  # 產生noise
    reward = np.zeros(n_scout)

    for j in range(0 , n_scout):
        # 對in_加一些擾動，得到pred
        pred = in_ + sigma * noise[j]

        # 利用pred得到reward，也就是計算pred與target有多少差距
        reward[j] = f(pred)

    # 對獎勵做normalization
    normalized_reward = (reward - np.mean(reward)) / np.std(reward)

    #                                                           50
    # 以normalized_reward當作權重，對所有noise[i , :]做加權總合 => ∑(noise[i , :] * normalized_reward[i])
    #                                                           i=0
    noise_weghted_sum = np.dot(noise.T , normalized_reward)

    # 更新in_
    in_ = in_ + alpha / (n_scout * sigma) * noise_weghted_sum

'''
總人口 : 5個
總共w的維度 : 3

(normalized_reward)_5x1 = [r1  =>  第1個人所產生的noise
                           r2  =>  第2個人所產生的noise
                           r3  =>  第3個人所產生的noise
                           r4  =>  第4個人所產生的noise
                           r5] =>  第5個人所產生的noise

(noise)_5x3 = [n11 , n12 , n13  =>  第1個人所產生的noise
               n21 , n22 , n23  =>  第2個人所產生的noise
               n31 , n32 , n33  =>  第3個人所產生的noise
               n41 , n42 , n43  =>  第4個人所產生的noise
               n51 , n52 , n53] =>  第5個人所產生的noise

(noise).T_3x5 = [n11 , n21 , n31 , n41 , n51
                 n12 , n22 , n32 , n42 , n52
                 n13 , n23 , n33 , n43 , n53]

(noise).T * normalized_reward = [n11*r1 + n21*r2 + n31*r3 + n41*r4 + n51*r5
                                 n12*r1 + n22*r2 + n32*r3 + n42*r4 + n52*r5
                                 n13*r1 + n23*r2 + n33*r3 + n43*r4 + n53*r5]

假設r3與r4最大，代表第3與第4個人所產生的noise占有最大的權重
所以第3與第4個人所產生的noise應該會是比較好的方向，可以帶領w得到最佳的值

'''





