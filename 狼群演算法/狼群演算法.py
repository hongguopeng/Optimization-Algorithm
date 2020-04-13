import numpy as np
import matplotlib
import matplotlib.pyplot as plt

WOLF_NUM = 200                 # 狼群數量
SCOUT_WOLF_NUM = 40            # 探狼數量，本來是在[WOLF_NUM/(α+1) , WOLF_NUM/α]這個區間隨機取值，但這裡先設定為constant，而α為探狼比例因子
SCOUT_STEP = 10                # 探狼隨機遊走的方向的數目
TMAX = 100                     # 探狼隨機有走的次數
DIM = 2                        # 2個維度
N_GENERATION = 200             # 共迭代100次
position_bound = [-100 , 100]  # 限制position的範圍
step_a = 2                     # 探狼遊走行為的步長
step_b = 4                     # 猛狼奔襲行為的步長
step_c = 1                     # 猛狼圍攻行為的步長
attack_factor = 5              # 距離判定因子，用來計算圍攻距離
distance_near = (position_bound[1] - position_bound[0]) / attack_factor # 猛狼位置與頭狼位置的距離小於此圍攻距離，猛狼即進行圍攻行為
DIEOUT_WOLF_NUM = 20           # 最後要淘汰狼群的數量，本來是在[WOLF_NUM/(β+1) , WOLF_NUM/β]這個區間隨機取值，但這裡先設定為constant，而β為群體更新比例因子

#--------------------------主要函數--------------------------#
# food function : 找3 * (1 - x_1)**2 * np.exp(-x_1**2 - (x_2 + 1)**2) - 10 * (x_1/5 - x_1**3 - x_2**5) * np.exp(-x_1**2 - x_2**2) - 1/3 * np.exp(-(x_1 + 1)**2 - x_2**2)的最大值
def get_fitness(position):
    wolf_num = position.shape[0]
    food = np.zeros(wolf_num)
    for i in range(0 , wolf_num):
        x_1 = position[i , 0]
        x_2 = position[i , 1]
        food[i] = 3 * (1 - x_1)**2 * np.exp(-x_1**2 - (x_2 + 1)**2) - 10 * (x_1/5 - x_1**3 - x_2**5) * np.exp(-x_1**2 - x_2**2) - 1/3 * np.exp(-(x_1 + 1)**2 - x_2**2)
    return food

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

# 初始化father_position、food、food_lead、index_lead、position_lead
# 初始化best_food、best_position
position = np.random.uniform(position_bound[0] , position_bound[1] , [WOLF_NUM , DIM])
food = get_fitness(position)
food_lead = food.max()
index_lead = food.argmax()
position_lead = position[index_lead , :]
best_food = -np.inf
best_position = position[0 , :].copy()
for generation in range(0 , N_GENERATION):
    # 從狼群中隨機決定擔任探狼的index，並得到探狼的位置
    scout_wolf_index = np.random.choice(np.arange(WOLF_NUM) ,
                                         size = SCOUT_WOLF_NUM ,
                                         replace = False)
    scout_wolf_position = position[scout_wolf_index  , :].copy()

    # 1.遊走行為
    # 探狼開始隨機遊走搜索獵物，若發現某個位置的食物量大於頭狼的食物量，更新頭狼位置，同時頭狼發出召喚行為
    # 若未發現，探狼繼續遊走直到達到最大遊走次數，頭狼在原本的位置發出召喚行為
    for tmax in range(0 , TMAX):
        food_scout = np.zeros(SCOUT_WOLF_NUM)
        # 每一隻探狼開始隨機遊走搜索獵物
        for scout_wolf in range(0 , SCOUT_WOLF_NUM):
            for scout_step in range(0 , SCOUT_STEP):
                scout_wolf_self = scout_wolf_position[scout_wolf , :].copy()
                scout_wolf_self = scout_wolf_self.reshape([1 , -1])

                # 探狼遊走到的位置scout_position
                scout_position = scout_wolf_position[scout_wolf , :] +\
                                np.sin(2 * np.pi * scout_step / SCOUT_STEP) * step_a
                scout_position = scout_position.reshape([1 , -1])
                # 讓scout_position在(position_bound[0] , position_bound[1])的範圍內
                scout_position = np.clip(scout_position , position_bound[0] , position_bound[1])

                # 若探狼遊走到的位置scout_position所發現的食物量大於上一個位置的食物量，則將scout_position當作現在探狼的位置
                if get_fitness(scout_position) > get_fitness(scout_wolf_self):
                    scout_wolf_position[scout_wolf , :] = scout_position

        # 計算現在探狼位置得到的食物量
        food_scout = get_fitness(scout_wolf_position)

        # 觀察有探狼位置的食物量中是否大於頭狼位置的食物量
        # 只要有大於的情況則將頭狼位置換成該探狼位置，並停止迴圈進入召喚行為
        food_scout_max = food_scout.max()
        if food_scout_max > food_lead:
            index_lead = food_scout.argmax()
            position_lead = scout_wolf_position[index_lead , :].copy()
            food_lead = food_scout_max
            break

    # 將一開始選為探狼的position替換為探狼進行遊走行為最後的位置
    # 並更新position_lead、food_lead
    position[scout_wolf_index , :] = scout_wolf_position.copy()
    food = get_fitness(position)
    index_lead = food.argmax()
    position_lead = position[index_lead , :].copy()
    food_lead = food.max()

    # 2.召喚行為
    # 每一隻狼會變成猛狼，並由頭狼召喚每一隻猛狼以較大的步長往頭狼位置奔襲
    # 若奔襲途中猛狼位置的食物量大於頭狼位置的食物量，則將對頭狼位置進行更新
    # 否則猛狼將繼續奔襲直到進入圍攻範圍
    flag = 0
    while flag == 0:
        for wolf in range(0 , WOLF_NUM):
            # 每一隻猛狼以較大的步長奔襲至頭狼的位置
            if wolf != index_lead:
                # (abs(position_lead - position[wolf , :]) + 1e-6) => 避免分母為0
                position[wolf , :] = position[wolf , :] +\
                                     step_b * (position_lead - position[wolf , :]) / (abs(position_lead - position[wolf , :]) + 1e-6)
                # 讓position[wolf , :]在(position_bound[0] , position_bound[1])的範圍內
                position[wolf , :] = np.clip(position[wolf , :] , position_bound[0] , position_bound[1])

                # 檢查現在猛狼位置的食物量如果大於頭狼位置的食物量
                # 將對頭狼位置進行換成目前猛狼位置，並重新發起召喚行為
                if get_fitness(position[wolf , :].reshape([1 , -1])) > food_lead:
                    position_lead = position[wolf , :].copy()
                    food_lead = get_fitness(position_lead.reshape([1 , -1]))
                    index_lead = wolf
                    break

                # 檢查現在猛狼的位置的食物量如果小於頭狼位置的食物量
                # 則計算猛狼目前位置與頭狼位置的距離
                if get_fitness(position[wolf , :].reshape([1 , -1])) <= food_lead:
                    distance_from_lead = abs(position_lead - position[wolf , :])
                    distance_from_lead = distance_from_lead.sum()

                    # 猛狼目前位置與頭狼位置的距離小於distance_near，猛狼以較小的步長就進行圍攻行為
                    if distance_from_lead <= distance_near:
                        λ = np.random.uniform(-1 , 1 , DIM)
                        siege_position = position[wolf , :] + step_c * λ * abs(position_lead - position[wolf , :])
                        siege_position = siege_position.reshape([1 , -1])
                        # 讓siege_position[wolf , :]在(position_bound[0] , position_bound[1])的範圍內
                        siege_position = np.clip(siege_position , position_bound[0] , position_bound[1])

                        if get_fitness(siege_position) > get_fitness(position[wolf , :].reshape([1 , -1])):
                            position[wolf , :] = siege_position

        # 若"每一隻"猛狼位置的食物量都小於頭狼位置的食物量，則結束召喚行為
        # 進入狼群更新機制
        if wolf == WOLF_NUM - 1:
            flag = 1

    # 3.狼群更新機制
    # 計算狼群中所有狼的位置的食物量
    # 淘汰掉DIEOUT_WOLF_NUM隻食物量最少的位置的狼，並隨機DIEOUT_WOLF_NUM隻新的狼
    # 有利於維護狼群的多樣性，使得不易陷入局部最優
    food = get_fitness(position)
    food_sort_index = food.argsort()[::-1]
    posotion = position[food_sort_index , :]
    position[-DIEOUT_WOLF_NUM: , :] = np.random.uniform(position_bound[0] , position_bound[1] , [DIEOUT_WOLF_NUM , DIM])

    # 3.狼群更新機制
    # 計算狼群中所有狼的位置的食物量
    # 淘汰掉DIEOUT_WOLF_NUM隻食物量最少的位置的狼，並隨機DIEOUT_WOLF_NUM隻新的狼
    # 有利於維護狼群的多樣性，使得不易陷入局部最優
    if food.max() > best_food:
        food_copy = food.copy()
        position_copy = position.copy()
        best_food = food_copy.max()
        best_food_index = food_copy.argmax()
        best_position = position_copy[best_food_index , :]

    print('Generation : {} , Best_Position : {} , Best_Food : {}'.format(generation , best_position , best_food))

    if 'sca' in globals(): sca.remove()
    sca = ax[0].scatter(position[: , 0] , position[: , 1] , alpha = 0.5 , s = 60 , c = 'black')
    plt.pause(0.01)

    color_count += 1
    if color_count == len(color_list): color_count = 0
    ax[1].scatter(best_position[0] , best_position[1] , s = 60 , color = color_list[color_count] , edgecolors = 'black')

