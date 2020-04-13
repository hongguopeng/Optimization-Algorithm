'''
加入交配、變種以及淘汰的概念，改善基本灰狼演算法易陷入局部最優等缺點
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

WOLF_NUM = 200                 # 狼群數量
DNA_SIZE = 2                   # 2個維度
N_GENERATION = 200             # 共迭代100次
group_bound = [-100 , 100]     # 限制group的範圍
CROSS_RATE = 0.8               # 狼群交配的機率
MUTATION_RATE = 0.1            # 每隻狼變種的機率
SCALE_BOUND = [0.2 , 0.8]      # 限制缩放因子的範圍

#--------------------------主要函數--------------------------#
# fitness function : 找3 * (1 - x_1)**2 * np.exp(-x_1**2 - (x_2 + 1)**2) - 10 * (x_1/5 - x_1**3 - x_2**5) * np.exp(-x_1**2 - x_2**2) - 1/3 * np.exp(-(x_1 + 1)**2 - x_2**2)的最大值
def get_fitness(group):
    wolf_num = group.shape[0]
    fitness = np.zeros(wolf_num)
    for i in range(0 , wolf_num):
        x_1 = group[i , 0]
        x_2 = group[i , 1]
        fitness[i] = 3 * (1 - x_1)**2 * np.exp(-x_1**2 - (x_2 + 1)**2) - 10 * (x_1/5 - x_1**3 - x_2**5) * np.exp(-x_1**2 - x_2**2) - 1/3 * np.exp(-(x_1 + 1)**2 - x_2**2)
    return fitness


def crossover(child , father_group):
    # 決定father_group中哪隻狼要當作mother_wolf與father交換基因
    # 如果crossover_index等於wolf，則繼續迴圈，直到crossover_index不等於wolf
    while True:
        crossover_index = np.random.randint(0 , WOLF_NUM , size = 1)
        if crossover_index != wolf:
            break
    mother_wolf = father_group[crossover_index , :].reshape([-1 , ])

    # 決定mother_wolf有幾條DNA要進行交換(crossover_num)
    # 並決定mother_wolf的哪幾條DNA要進行交換(cross_points)
    crossover_num = np.random.randint(1 , DNA_SIZE + 1 , size = 1)[0]
    cross_points = np.random.choice(np.arange(DNA_SIZE) , size = crossover_num , replace = False)

    # 將child指定的基因位置換成mother_wolf指定的基因位置的基因
    crossover_decision = np.random.rand()
    if crossover_decision < CROSS_RATE:
        child[cross_points] = mother_wolf[cross_points]

    return child


def mutate(child , father_group):
    # 決定father_group中隨機挑出3隻狼，並配合縮放因子，產生mutant_wolf
    # 如果index_a、index_b、index_c等於wolf，則繼續迴圈，直到index_a、index_b、index_c不等於wolf
    while True:
        index_a = np.random.randint(0 , WOLF_NUM)
        index_b = np.random.randint(0 , WOLF_NUM)
        index_c = np.random.randint(0 , WOLF_NUM)
        if index_a != wolf and index_b != wolf and index_c != wolf:
            break
    scale_factor = np.random.uniform(SCALE_BOUND[0] , SCALE_BOUND[1] , [1 , DNA_SIZE]) # 隨機產生缩放因子
    # 以差分進化的方式決定mutant_wolf
    mutant_wolf = father_group[index_a , :] +\
                  scale_factor * (father_group[index_b , :] - father_group[index_c , :])
    mutant_wolf = mutant_wolf.reshape([-1 , ])

    # 決定有幾條DNA要進行變種(mutant_num)
    # 並決定child要繼承mutant_wolf的哪幾條DNA要(mutant_points)
    mutant_num = np.random.randint(1 , DNA_SIZE + 1 , size = 1)[0]
    mutant_points = np.random.choice(np.arange(DNA_SIZE) , size = mutant_num , replace = False)

    # 將child指定的基因位置換成mutant_wolf指定的基因位置的基因
    mutant_decision = np.random.rand()
    if mutant_decision < MUTATION_RATE:
        child[mutant_points] = mutant_wolf[mutant_points]

    return child


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

# 初始化father_group、fitness、best_fitness、best_group
father_group = np.random.uniform(group_bound[0] , group_bound[1] , [WOLF_NUM , DNA_SIZE])
best_fitness = -np.inf
best_group = father_group[0 , :].copy()

for generation in range(0 , N_GENERATION):

    # 根據適應度決定α狼(其適應度最佳、離最優值最接近的狼)、β狼與δ狼(適應度次佳的兩個個體)，剩下的則為ω狼
    fitness = get_fitness(father_group)
    sort_index = fitness.argsort()[::-1]
    α_wolf = father_group[sort_index[0] , :].copy()
    β_wolf = father_group[sort_index[1] , :].copy()
    δ_wolf = father_group[sort_index[2] , :].copy()

    # a會越來越小，隨著generation的增加，a會從2慢慢以線性的方式減少到0
    # 慢慢縮小範圍的感覺
    a = 2 - generation * (2 / N_GENERATION)

    for wolf in range(0 , WOLF_NUM):
        ω_wolf = father_group[wolf , :]

        r1 = np.random.rand(DNA_SIZE)
        r2 = np.random.rand(DNA_SIZE)
        A1 = a * (2 * r1 - 1) # 讓A1在[-a , a]之間
        C1 = 2 * r2 # C1在[0 , 2]之間
        # 計算α狼與ω狼之間的距離
        distance_α = abs(C1 * α_wolf - ω_wolf)
        # ω狼根據α狼更新的位置X1，可以想成是ω狼往α狼的方向靠近
        X1 = α_wolf - A1 * distance_α

        r1 = np.random.rand(DNA_SIZE)
        r2 = np.random.rand(DNA_SIZE)
        A2 = a * (2 * r1 - 1) # 讓A1在[-a , a]之間
        C2 = 2 * r2 # C3在[0 , 2]之間
        # 計算β狼與ω狼之間的距離
        distance_β = abs(C2 * β_wolf - ω_wolf)
        # ω狼根據β狼更新的位置X2，可以想成是ω狼往β狼的方向靠近
        X2 = β_wolf - A2 * distance_β

        r1 = np.random.rand(DNA_SIZE)
        r2 = np.random.rand(DNA_SIZE)
        A3 = a * (2 * r1 - 1) # 讓A1在[-a , a]之間
        C3 = 2 * r2 # C3在[0 , 2]之間
        # 計算δ狼與ω狼之間的距離
        distance_δ = abs(C3 * δ_wolf - ω_wolf)
        # ω狼根據δ狼更新的位置X3，可以想成是ω狼往δ狼的方向靠近
        X3 = δ_wolf - A3 * distance_δ

        # ω狼權衡α狼、β狼、δ狼所更新的位置new_wolf
        # 這裡是用直接平均的方式，當然也可以用加權平均的方式計算new_wolf
        new_wolf = (X1 + X2 + X3) / 3
        # 讓new_wolf在(group_bound[0] , group_bound[1])的範圍內
        father_group[wolf , :] = np.clip(new_wolf , group_bound[0] , group_bound[1])

    # 先讓child_group繼承father_group全部的基因
    child_group = father_group.copy()
    for wolf in range(0 , WOLF_NUM):
        child = child_group[wolf , :].copy()
        child = crossover(child , father_group)  # 執行交配
        child = mutate(child, father_group)      # 執行變種
        child_group[wolf , :] = child

    # 讓child_group在(group_bound[0] , group_bound[1])的範圍內
    child_group = np.clip(child_group , group_bound[0] , group_bound[1])

    #------------------------不適者淘汰------------------------#
    # 把child_group適應度大於ather_group適應度的index抓出來 => better_index
    # 將father_group[better_index , :]換成child_group[better_index , :]
    father_fitness = get_fitness(father_group)
    child_fitness = get_fitness(child_group)
    better_index = child_fitness > father_fitness
    father_group[better_index , :] = child_group[better_index , :]
    #------------------------不適者淘汰------------------------#

    if fitness.max() > best_fitness:
        fitness_copy = fitness.copy()
        father_group_copy = father_group.copy()
        best_fitness = fitness_copy.max()
        best_fitness_index = fitness_copy.argmax()
        best_group = father_group_copy[best_fitness_index , :]

    print('Generation : {} , Best_Group : {} , Best_Fitness : {:.2f}'.format(generation , best_group , best_fitness))

    if 'sca' in globals(): sca.remove()
    sca = ax[0].scatter(father_group[: , 0] , father_group[: , 1] , alpha = 0.5 , s = 60 , c = 'black')
    plt.pause(0.001)

    color_count += 1
    if color_count == len(color_list): color_count = 0
    ax[1].scatter(best_group[0] , best_group[1] , s = 60 , color = color_list[color_count] , edgecolors = 'black')
