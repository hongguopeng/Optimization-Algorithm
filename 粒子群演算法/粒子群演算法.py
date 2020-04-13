import numpy as np
import matplotlib
import matplotlib.pyplot as plt

PARTICLE_NUM = 100             # particle的數目
DIM = 2                        # 2個維度
particle_bound = [-100 , 100]  # 限制particle的範圍
speed_bound = [-20 , 20]       # 限制speed的範圍
N_GENERATION = 500             # 共迭代500次

# 慣性權重較大時，PSO傾向於開發者，可以大範圍地探索新的區域
# 慣性權重較小時，PSO傾向於探測者，在局部範圍內找尋最佳值
weight_start , weight_end = 0.9 , 0.4
WEIGHT = weight_start

# C1和C2稱為學習因子
# 可以讓particle向個體的最好的particle靠近，
# 也讓所有particle中最好的particle靠近
# 通常取C1 = C2 = 2
C1 , C2 = 2 , 2


#--------------------------主要函數--------------------------#
# fitness function : 找-(x_1^2 + 2*x_2^2 - 0.3*cos(3*π*x_1) - 0.4*cos(4*π*x_2) + 0.7)的最大值
def get_fitness(particle):
    x_1 = particle[: , 0]
    x_2 = particle[: , 1]
    π = np.pi
    return -(x_1**2 + 2*x_2**2 - 0.3*np.cos(3*π*x_1) - 0.4*np.cos(4*π*x_2) + 0.7)


# 把新的適應度大於原來適應度的index抓出來 => better_index
# local_best_particle[better_index , 換成new_particle[better_index , :]
def find_local_best(local_best_particle , new_particle , new_fitness , old_fitness):
    better_index = new_fitness > old_fitness
    local_best_particle[better_index , :] = new_particle[better_index , :]
    return local_best_particle


# 找到所有particle中最好的particle
def find_global_best(old_particle , new_particle , old_fitness , new_fitness):
    # 計算原來particle中最好的particle與適應度
    old_global_best_particle_index = old_fitness.argmax()
    old_global_best_particle = old_particle[old_global_best_particle_index , :]

    # 計算新的particle中最好的particle與適應度
    new_global_best_particle_index = new_fitness.argmax()
    new_global_best_particle = new_particle[new_global_best_particle_index , :]

    # 假如新的適應度中的最大值大於原來適應度中的最大值，即可將最好的particle換掉
    if new_fitness.max() > old_fitness.max():
        return new_global_best_particle
    else:
        return old_global_best_particle


# 畫等高線圖
def contour(ax):
    n = 500
    x = np.linspace(-100 , 100 , n)
    Y , X = np.meshgrid(x , x)
    Z = np.zeros_like(X)
    for i in range(0 , n):
        for j in range(0 , n):
            Z[i , j] = get_fitness(np.array([[x[i] , x[j]]]))
    ax[0].contourf(X , Y , Z , 100 , alpha = 0.75 , cmap = plt.cm.rainbow)
    C = ax[0].contour(X , Y , Z , 10 , alpha = 0.75 , colors = 'black')
    ax[0].clabel(C , inline = True , fontsize= 10)
    ax[0].set_ylim(-100 , 100)
    ax[0].set_xlim(-100 , 100)
#--------------------------主要函數--------------------------#

# 總共畫兩張圖
fig , ax = plt.subplots(1 , 2 , figsize = (20 , 6))
color_list = list(matplotlib.colors.cnames.values()) # 顏色列表
color_count = 0
# 畫等高線圖
contour(ax)
plt.ion()

# 初始化particle
particle = np.random.uniform(-100 , 100 , [PARTICLE_NUM , DIM])

# 初始化speed
speed = np.random.uniform(-20 , 20 , [PARTICLE_NUM , DIM])

# 初始化每個個體的最好的particle
local_best_particle = particle

# 初始化所有particle最好的particle
fitness = get_fitness(particle)
global_best_particle_index = fitness.argmax()
global_best_particle = particle[global_best_particle_index , :].copy()

for generation in range(0 , N_GENERATION):

    # 隨機生成r1與r2
    r1 , r2 = np.random.rand() , np.random.rand()

    # 生成新的speed，並讓speed在(speed_bound[0] , speed_bound[1])的範圍內
    # 新的speed = 上個時間的speed + 個體認為的最佳方向 + 群體認為的最佳方向
    speed = WEIGHT * speed +\
            C1 * r1 * (local_best_particle - particle) +\
            C2 * r2 * (global_best_particle - particle)
    speed = np.clip(speed , speed_bound[0] , speed_bound[1])

    # 生成new_particle，並讓new_particle在(particle_bound[0] , particle_bound[1])的範圍內
    new_particle = particle + speed
    new_particle = np.clip(new_particle , particle_bound[0] , particle_bound[1])

    # 計算原來particle的適應度old_fitness
    # 計算新的particle的適應度new_fitness
    old_fitness = get_fitness(particle)
    new_fitness = get_fitness(new_particle)

    # 新的適應度與原來適應度的差異小於TOL，即可停止迴圈
    if abs(new_fitness.max() - old_fitness.max()) <= 1e-8:
        break

    # 把新的適應度大於原來適應度的index抓出來，以更新個別particle的最好的particle
    local_best_particle = find_local_best(local_best_particle , new_particle , new_fitness , old_fitness)

    # 找到所有particle中最好的particle
    global_best_particle = find_global_best(particle , new_particle , old_fitness , new_fitness)

    # 更新particle
    particle = new_particle

    # 線性減少慣性權重，則PSO在一開始時具有良好的全域搜索能力
    # 能夠迅速定位到接近全域最佳解的區域，而在後期具有良好的局部搜索能力
    WEIGHT = weight_start - (weight_start - weight_end) / N_GENERATION * (generation + 1)

    print('Generation : {} , Best_fitness : {:.2f} , Global_Best_Particle : {}'.format(generation , old_fitness.max() , global_best_particle))

    if 'sca' in globals(): sca.remove()
    sca = ax[0].scatter(particle[: , 0] , particle[: , 1] , alpha = 0.5 , s = 60 , c = 'black')

    # 可以看到隨著每一次迭代，global_best_particle會越來越集中在一個地方
    color_count += 1
    if color_count == len(color_list): color_count = 0
    ax[1].scatter(global_best_particle[0] , global_best_particle[1] , s = 60 , color = color_list[color_count] , edgecolors = 'black')
    plt.pause(0.1)

plt.ioff()
plt.show()