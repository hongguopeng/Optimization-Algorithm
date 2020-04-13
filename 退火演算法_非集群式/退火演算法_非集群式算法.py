import numpy as np
import matplotlib
import matplotlib.pyplot as plt

TEMPERATURE = 1000            # 初始溫度
position_bound = [-20 , 20]   # 限制position的範圍
DIM = 2                       # 2個維度
ITERATION = 200

#--------------------------主要函數--------------------------#
# fitness function : 找3 * (1 - x_1)**2 * np.exp(-x_1**2 - (x_2 + 1)**2) - 10 * (x_1/5 - x_1**3 - x_2**5) * np.exp(-x_1**2 - x_2**2) - 1/3 * np.exp(-(x_1 + 1)**2 - x_2**2)的最大值
# 但是退火演算法是專門求最小值的方法，所以最後加1個負號
def get_fitness(position):
    bee_num = position.shape[0]
    fitness = np.zeros(bee_num)
    for i in range(0 , bee_num):
        x_1 = position[i , 0]
        x_2 = position[i , 1]
        fitness[i] = 3 * (1 - x_1)**2 * np.exp(-x_1**2 - (x_2 + 1)**2) - 10 * (x_1/5 - x_1**3 - x_2**5) * np.exp(-x_1**2 - x_2**2) - 1/3 * np.exp(-(x_1 + 1)**2 - x_2**2)
    return -fitness


# 畫等高線圖
def contour(ax):
    n = 300
    x = np.linspace(-5 , 5 , n)
    Y , X = np.meshgrid(x , x)
    Z = np.zeros_like(X)
    for i in range(0 , n):
        for j in range(0 , n):
            Z[i , j] = get_fitness(np.array([[x[i] , x[j]]]))
    ax[0].contourf(X , Y , Z , 50 , alpha = 0.75 , cmap = plt.cm.rainbow)
    C = ax[0].contour(X , Y , Z , 10 , alpha = 0.75 , colors = 'black')
    ax[0].clabel(C , inline = True , fontsize = 10)
    ax[0].set_ylim(-5 , 5)
    ax[0].set_xlim(-5 , 5)
#--------------------------主要函數--------------------------#

# 總共畫兩張圖
fig , ax = plt.subplots(1 , 2 , figsize = (20 , 6))
color_list = list(matplotlib.colors.cnames.values()) # 顏色列表
color_count = 0
plt.ion()
# 畫等高線圖
contour(ax)
plt.ion()

# 初始化position、value、best_value、best_position、generation
position = np.random.uniform(position_bound[0] , position_bound[1] , [1 , DIM])
best_value = np.inf
best_position = position.copy()
generation = 0

while True:
    generation += 1

    for i in range(0 , ITERATION):
        # 生成new_position，並讓new_position在(position_bound[0] , position_bound[1])的範圍內
        new_position = position + (2 * np.random.rand(1 , DIM) - 1)
        new_position = np.clip(new_position , position_bound[0] , position_bound[1])

        # 計算position與new_position的適應度
        value = get_fitness(position)[0]
        new_value = get_fitness(new_position)[0]

        # 計算增量ΔT = get_fitness(new_position)[0] - get_fitness(position)[0]
        # situation 1 : ΔT小於0，則讓new_position等於position
        # situation 2 : ΔT大於0，以概率exp(-ΔT/T)來決定讓position要不要等於new_position
        ΔT = new_value - value
        if ΔT < 0:
            position = new_position
        elif ΔT > 0:
            prob = np.exp(-ΔT / TEMPERATURE)
            if prob > np.random.rand():
                position = new_position

    # 讓TEMPERATURE逐漸減少，讓結果可以逐漸收斂
    TEMPERATURE *= 0.99
    if TEMPERATURE <= 0.001:
        break

    if value < best_value:
        best_value = value
        best_position = position

    print('Generation : {} , Best_Position : {} ,  Best_Value : {}'.format(generation , best_position[0] , best_value))

    ax[0].scatter(position[: , 0] , position[: , 1] , alpha = 0.5 , s = 60 , c = 'red' , edgecolors = 'black')
    # plt.pause(0.000001)

    color_count += 1
    if color_count == len(color_list): color_count = 0
    ax[1].scatter(best_position[0 , 0] , best_position[0 , 1] , s = 60 , color = color_list[color_count] , edgecolors = 'black')

plt.ioff()
plt.show()
