import numpy as np
import matplotlib
import matplotlib.pyplot as plt

SCOUT_BEE_NUM = 100            # 斥候蜂數量
GREEDY_BEE_NUM = 100           # 貪婪蜂數量
INSPECT_BEE_NEM = 100          # 考察蜂數量
DIM = 2                        # 2個維度
N_GENERATION = 100             # 共迭代100次
LIMIT = int(0.6 * DIM * SCOUT_BEE_NUM)  # 一個position最多採集次數，若超過LIMIT，則放棄該position
position_bound = [-50 , 50]      # 限制position的範圍

#--------------------------主要函數--------------------------#
# fitness function : 找3 * (1 - x_1)**2 * np.exp(-x_1**2 - (x_2 + 1)**2) - 10 * (x_1/5 - x_1**3 - x_2**5) * np.exp(-x_1**2 - x_2**2) - 1/3 * np.exp(-(x_1 + 1)**2 - x_2**2)的最大值
def get_fitness(position):
    bee_num = position.shape[0]
    fitness = np.zeros(bee_num)
    for i in range(0 , bee_num):
        x_1 = position[i , 0]
        x_2 = position[i , 1]
        fitness[i] = 3 * (1 - x_1)**2 * np.exp(-x_1**2 - (x_2 + 1)**2) - 10 * (x_1/5 - x_1**3 - x_2**5) * np.exp(-x_1**2 - x_2**2) - 1/3 * np.exp(-(x_1 + 1)**2 - x_2**2)
    return fitness


# 輪盤法
def dart_method(prob):
    dart = np.random.rand(1)[0]
    indicator = dart
    count = 0
    while indicator > 0:
        indicator -= prob[count]
        count += 1
    return count - 1


# 畫等高線圖
def contour(ax):
    n = 300
    x = np.linspace(-3 , 3 , n)
    Y , X = np.meshgrid(x , x)
    Z = np.zeros_like(X)
    for i in range(0 , n):
        for j in range(0 , n):
            Z[i , j] = get_fitness(np.array([[x[i] , x[j]]]))
    ax[0].contourf(X , Y , Z , 50 , alpha = 0.75 , cmap = plt.cm.rainbow)
    C = ax[0].contour(X , Y , Z , 10 , alpha = 0.75 , colors = 'black')
    ax[0].clabel(C , inline = True , fontsize = 10)
    ax[0].set_ylim(-3 , 3)
    ax[0].set_xlim(-3 , 3)
#--------------------------主要函數--------------------------#

# 總共畫兩張圖
fig , ax = plt.subplots(1 , 2 , figsize = (20 , 6))
color_list = list(matplotlib.colors.cnames.values()) # 顏色列表
color_count = 0
# 畫等高線圖
contour(ax)
plt.ion()

# 初始化position、honey_amount、bad_position_freq
position = np.random.uniform(position_bound[0] , position_bound[1] , [SCOUT_BEE_NUM , DIM])
honey_amount = get_fitness(position)        # 每個position的花蜜量
bad_position_freq = np.zeros(SCOUT_BEE_NUM) # 紀錄採集的花量比前一次還要差的次數

for generation in range(0 , N_GENERATION):
    # 1. 斥候蜂階段
    for scout_bee in range(0 , SCOUT_BEE_NUM):
        # 隨機選擇生成choose，但不能跟scout_bee相同
        while True:
            choose = np.random.randint(0 , SCOUT_BEE_NUM , size = 1)
            if choose != scout_bee:
                break

        # 生成new_position，並讓new_position在(position_bound[0] , position_bound[1])的範圍內
        Φ = np.random.uniform(-1 , 1 , [1 , DIM]) # 隨機生成範圍在[-1 , 1]間的係數Φ
        new_position = position[scout_bee , :] + Φ * (position[scout_bee , :] - position[choose , :])
        new_position = np.clip(new_position , position_bound[0] , position_bound[1])

        # 根據new_position，派出斥候蜂去new_position採集花蜜，採集到花蜜量new_honey_amount
        new_honey_amount = get_fitness(new_position)

        # 若在new_position採集到的花蜜量new_honey_amount大於原本花蜜量honey_amount[scout_bee]
        # 則將honey_amount[scout_bee]替換成new_honey_amount
        # 若在new_position採集到的花蜜量new_honey_amount小於原本花蜜量honey_amount[scout_bee]
        # 則bad_position_freq[scout_bee]加1
        if new_honey_amount > honey_amount[scout_bee]:
            position[scout_bee , :] = new_position
            honey_amount[scout_bee] = new_honey_amount
        else:
            bad_position_freq[scout_bee] += 1
    
    # 根據斥候蜂更新的花蜜量，計算機率分佈
    honey_prob = np.exp(-honey_amount / honey_amount.mean())
    honey_prob = honey_prob / honey_prob.sum()
    
    # 2. 貪婪蜂階段
    for greedy_bee in range(0 , GREEDY_BEE_NUM):
        # 用前面計算的機率分佈與輪盤法選擇position
        dart = dart_method(honey_prob)
        # 如果覺得麻煩，也可以直接用np.random.choice
        # dart = np.random.choice(np.arange(GREEDY_BEE_NUM) ,
        #                         size = 1 ,
        #                         p = honey_prob)

        # 隨機選擇生成choose，但不能跟dart相同
        while True:
            choose = np.random.randint(0 , SCOUT_BEE_NUM , size = 1)
            if choose != dart:
                break
        # 生成new_position，並讓new_position在(position_bound[0] , position_bound[1])的範圍內
        Φ = np.random.uniform(-1 , 1 , [1 , DIM]) # 隨機生成範圍在[-1 , 1]間的係數Φ
        new_position = position[dart , :] + Φ * (position[dart , :] - position[choose , :])
        new_position = np.clip(new_position , position_bound[0] , position_bound[1])

        # 根據new_position，派出貪婪蜂去new_position採集花蜜，採集到花蜜量new_honey_amount
        new_honey_amount = get_fitness(new_position)

        # 若在new_position採集到的花蜜量new_honey_amount大於原本花蜜量honey_amount[dart]
        # 則將honey_amount[dart]替換成new_honey_amount
        # 若在new_position採集到的花蜜量new_honey_amount小於原本花蜜量honey_amount[dart]
        # 則bad_position_freq[dart]加1
        if new_honey_amount > honey_amount[dart]:
            position[dart , :] = new_position
            honey_amount[dart] = new_honey_amount
        else:
            bad_position_freq[dart] += 1

    # 3. 考察蜂
    for inspect_bee in range(0 , INSPECT_BEE_NEM):
        # 考察蜂負責考察bad_position_freq中每個值是否大於LIMIT
        # 若bad_position_freq[i]大於LIMIT
        # 則放棄position[i , :]，並重新找一個new_position塞到position[i , :]
        # 並把該new_position所得到的花蜜量new_honey_amount塞到honey_amount[i]中
        # 最後把bad_position_freq[i]歸零，重新開始計算
        if bad_position_freq[inspect_bee] >= LIMIT:
            new_position = np.random.uniform(position_bound[0] , position_bound[1] , [1 , DIM])
            new_position = np.clip(new_position , position_bound[0] , position_bound[1])
            position[inspect_bee , :] = new_position
            new_honey_amount = get_fitness(new_position)
            honey_amount[inspect_bee] = new_honey_amount
            bad_position_freq[inspect_bee] = 0
    
    best_honey_aomunt = honey_amount.max()
    best_honey_aomunt_index = honey_amount.argmax()
    best_position = position[best_honey_aomunt_index , :]
    print('Generation : {} , Best_Position : {} ,  Best_Honey_Aomunt : {:.2f}'.format(generation , best_position , best_honey_aomunt))

    if 'sca' in globals(): sca.remove()
    sca = ax[0].scatter(position[: , 0] , position[: , 1] , alpha = 0.5 , s = 60 , c = 'black')
    plt.pause(0.1)

    # 可以看到隨著每一次迭代，best_position會越來越集中在一個地方
    color_count += 1
    if color_count == len(color_list): color_count = 0
    ax[1].scatter(best_position[0] , best_position[1] , s = 60 , color = color_list[color_count] , edgecolors = 'black')
    plt.pause(0.1)

plt.ioff()
plt.show()