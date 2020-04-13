import numpy as np
import matplotlib
import matplotlib.pyplot as plt

FISH_NUM = 100               # 魚群數量
DIM = 2                      # 2個維度
N_GENERATION = 200           # 共迭代200次
TRY_NUMBER = 100             # 最多嘗試次數
position_bound = [-20 , 20]  # 限制position的範圍
VISUAL = 1                   # 可視範圍
DELTA = 0.618                # 群聚因子
STEP = 0.1                   # 步長

#--------------------------主要函數--------------------------#
# fitness function : 找3 * (1 - x_1)**2 * np.exp(-x_1**2 - (x_2 + 1)**2) - 10 * (x_1/5 - x_1**3 - x_2**5) * np.exp(-x_1**2 - x_2**2) - 1/3 * np.exp(-(x_1 + 1)**2 - x_2**2)的最大值
def get_fitness(position):
    fish_num = position.shape[0]
    fitness = np.zeros(fish_num)
    for i in range(0 , fish_num):
        x_1 = position[i , 0]
        x_2 = position[i , 1]
        fitness[i] = 3 * (1 - x_1)**2 * np.exp(-x_1**2 - (x_2 + 1)**2) - 10 * (x_1/5 - x_1**3 - x_2**5) * np.exp(-x_1**2 - x_2**2) - 1/3 * np.exp(-(x_1 + 1)**2 - x_2**2)
    return fitness


# 收集position_self的可視範圍內的position[i , :]，與可視範圍內的position[i , :]的數量
def in_viual(position , position_self , fish):
    position_in_visual = []
    for i in range(0 , len(position)):
        if i != fish:
            x_diff = position[i , :][0] - position_self[0]
            y_diff = position[i , :][1] - position_self[0]
            distance = (x_diff ** 2 + y_diff ** 2) ** 0.5
            if distance < VISUAL:
                position_in_visual.append(position[i , :])
    return np.array(position_in_visual) , len(position_in_visual)


def fish_prey(position_self , food_self , position_bound , flag_random = True):
    for i in range(0 , TRY_NUMBER):
        # 覓食行為是先隨機尋找1個位置position_move
        position_move = position_self + STEP * (2 * np.random.rand(1 , DIM) - 1)
        position_move = np.clip(position_move , position_bound[0] , position_bound[1])
        food_move = get_fitness(position_move)

        # 若position_move的食物量food_move更多
        # 就往該方向去移動，而移動的向量是normalization的向量
        if food_move > food_self:
            flag_random = False
            vector = position_move - position_self
            length = np.linalg.norm(vector)
            normalized_vector = vector / length # 對vector作normalization

            # 生成position_next，並讓position_next在(position_bound[0] , position_bound[1])的範圍內
            position_next = position_self + STEP * np.random.rand() * normalized_vector
            position_next = np.clip(position_next , position_bound[0] , position_bound[1])
            break

    # 假如試了TRY_NUMBER次，food_move還是小於food_self，那就進入隨機行為
    if flag_random:
        # 生成position_next，並讓position_next在(position_bound[0] , position_bound[1])的範圍內
        position_next = position_self + STEP * (2 * np.random.rand(1 , DIM) - 1)
        position_next = np.clip(position_next , position_bound[0] , position_bound[1])
        food_next = get_fitness(position_next)

    food_next = get_fitness(position_next)

    return position_next , food_next


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
plt.ion()
# 畫等高線圖
contour(ax)
plt.ion()

# 初始化position、food、best_food、best_position
position = np.random.uniform(position_bound[0] , position_bound[1] , [FISH_NUM , DIM])
food = get_fitness(position) # 每個position的食物量
best_food = -np.inf
best_position = position[0 , :].copy()

for generation in range(0 , N_GENERATION):
    for fish in range(0 , FISH_NUM):
        # 計算每一條魚所在的position_self與該地的食物量
        position_self = position[fish , :]
        food_self = food[fish]
    
        # 收集position_self的可視範圍內的所有position，與可視範圍內所有position的數量
        position_in_visual , in_visual_num = in_viual(position , position_self , fish)
    
        # position_self可視範圍內一定要出現至少1個position，滿足這一個條件才能檢查是否能進入群聚行為
        if in_visual_num > 0:
            # 尋找當前position_self可視範圍內的所有position
            # 並對這些position取mean，當作中心位置position_in_visual_mean
            # 取出position_in_visual_mean的食物量food_in_visual_mean
            position_in_visual_mean = position_in_visual.mean(axis = 0)
            position_in_visual_mean = position_in_visual_mean.reshape([1 , -1]) # position_in_visual_mean的shape (2 , )=>(2 , 1)
            food_in_visual_mean = get_fitness(position_in_visual_mean)

            # position_in_visual_mean的食物量food_in_visual_mean更多 => (♠)
            # 在可視範圍內position的數量佔所有position的數量的比例要小於群聚因子 => (♥)
            # 滿足(♠)與(♥)這2個條件，才能進入群聚行為
            if food_in_visual_mean > food_self and in_visual_num / FISH_NUM < DELTA:
                # 往position_in_visual_mean方向來移動，而移動的向量是normalization的向量
                vector = position_in_visual_mean - position_self
                length = np.linalg.norm(vector)
                normalized_vector = vector / length # 對vector作normalization
    
                # 生成position_swarm，並讓position_swarm在(position_bound[0] , position_bound[1])的範圍內
                position_swarm = position_self + STEP * np.random.rand() * normalized_vector
                position_swarm = np.clip(position_swarm , position_bound[0] , position_bound[1])
                food_swarm = get_fitness(position_swarm)
    
            # 不符合群聚行為的2個條件，即進入覓食行為
            else:
                position_swarm , food_swarm = fish_prey(position_self , food_self , position_bound)
    
        # 在可視範圍內的position的數量如果等於0，即進入覓食行為
        else:
            position_swarm , food_swarm = fish_prey(position_self , food_self , position_bound)

        # position_self可視範圍內一定要出現至少1個position，滿足這一個條件才能檢查是否能進入追尾行為
        if in_visual_num > 0:
            # 尋找當前position_self可視範圍內的所有position中食物量最多position_in_visual_max
            # 以及position_in_visual_max的food_in_visual_max
            food_in_visual = get_fitness(position_in_visual)
            food_in_visual_max = food_in_visual.max()
            food_in_visual_max_index = food_in_visual.argmax()
            position_in_visual_max = position_in_visual[food_in_visual_max_index , :]
            position_in_visual_max = position_in_visual_max.reshape([1 , -1]) # position_in_visual_max的shape (2 , )=>(2 , 1)

            # position_in_visual_max的食物量food_in_visual_max更多 => (♦)
            # 在可視範圍內position的數量佔所有position的數量的比例要小於群聚因子 => (♣)
            # 滿足(♦)與(♣)這2個條件，才能進入追尾行為
            if food_in_visual_max > food_self and in_visual_num / FISH_NUM < DELTA:
                # 往position_in_visual_max方向來移動，而移動的向量是normalization的向量
                vector = position_in_visual_max - position_self
                length = np.linalg.norm(vector)
                normalized_vector = vector / length # 對vector作normalization
    
                # 生成position_follow，並讓position_follow在(position_bound[0] , position_bound[1])的範圍內
                position_follow = position_self + STEP * np.random.rand() * normalized_vector
                position_follow = np.clip(position_follow , position_bound[0] , position_bound[1])
                food_follow = get_fitness(position_follow)
    
            # 不符合追尾行為的2個條件，即進入覓食行為
            else:
                position_follow , food_follow = fish_prey(position_self , food_self , position_bound)
    
        # 在可視範圍內的position的數量等於0，即進入覓食行為
        else:
            position_follow , food_follow = fish_prey(position_self , food_self , position_bound)

        # 比較群聚行為與追尾行為得到的食物量
        if food_swarm >= food_follow:
            position[fish , :] = position_swarm
            food[fish] = food_swarm
        elif food_swarm < food_follow:
            position[fish , :] = position_follow
            food[fish] = food_follow

    if food.max() > best_food:
        food_copy = food.copy()
        position_copy = position.copy()
        best_food = food_copy.max()
        best_food_index = food_copy.argmax()
        best_position = position_copy[best_food_index , :]

    print('Generation : {} , Best_Position : {} ,  Best_Food : {:.2f}'.format(generation , best_position , best_food))

    if 'sca' in globals(): sca.remove()
    sca = ax[0].scatter(position[: , 0] , position[: , 1] , alpha = 0.5 , s = 60 , c = 'black')
    plt.pause(0.001)

    color_count += 1
    if color_count == len(color_list): color_count = 0
    ax[1].scatter(best_position[0] , best_position[1] , s = 60 , color = color_list[color_count] , edgecolors = 'black')

plt.ioff()
plt.show()
