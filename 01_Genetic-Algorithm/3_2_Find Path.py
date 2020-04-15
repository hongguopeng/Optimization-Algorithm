import matplotlib.pyplot as plt
import numpy as np

DNA_SIZE = 200        # DNA的長度(代表從起點走到終點總共只能走200步)
POP_SIZE = 100        # 總人口數(代表有100種走法)
CROSS_RATE = 0.8      # 交配的機率
MUTATE_RATE = 0.0001  # 變種的機率
N_GENERATIONS = 100   # 共迭代100次
RESOLUTION = 0.5      # 在translateDNA有說明

START_POINT = [0 , 5]  # 起點
GOAL_POINT = [10 , 5]  # 終點
INCREMENT_BOUND = [0 , 2] # 隨機生成0與1(後面會用到)
OBSTACLE_LINE = np.array([[5 , 2] , [5 , 8]])

#--------------------------主要函數--------------------------#
# pop_x => 記錄每一步在x方向的移動量(有正有負)
# pop_y => 記錄每一步在y方向的移動量(有正有負)
# location_x => 記錄每一步所經過的x的位置
# location_y => 記錄每一步所經過的y的位置
# (location_x[t_step] , location_y[t_step]) => t_step所在的(x , y)
def translateDNA(pop_x , pop_y):     # city_position 代表每一個城市的座標(x , y)
    pop_x = (pop_x - 0.5) * RESOLUTION # 減去0.5 => 可以讓 pop_x有正有負 ； 可以將RESOLUTION調小，讓每一步在x方向移動的距離小一點，但太小可能會走不到終點
    pop_y = (pop_y - 0.5) * RESOLUTION # 減去0.5 => 可以讓 pop_x有正有負 ； 可以將RESOLUTION調小，讓每一步在y方向移動的距離小一點，但太小可能會走不到終點
    pop_x[: , 0] = START_POINT[0]
    pop_y[: , 0] = START_POINT[1]
    location_x = np.cumsum(pop_x , axis = 1)
    location_y = np.cumsum(pop_y , axis = 1)
    return location_x , location_y


# 得到適應度，而適應度其實就是機率分布
# distance_goal代表最後所在座標與終點的距離
# distance_goal越大，適應度越低
def get_fitness(location_x , location_y):
    x_diff = GOAL_POINT[0] - location_x[: , -1]
    y_diff = GOAL_POINT[1] - location_y[: , -1]
    distance_goal = (x_diff ** 2 +  y_diff ** 2) ** 0.5
    fitness = np.power(1 / (distance_goal + 1) , 2)

    # x => [4 , 6] 與 y => [2 , 8]，這個範圍是防空識別區
    # 若(location_x[i , j] , location_y[i , j])出現在這麼範圍，fitness[i]會得到極低的值
    # 代表i這個走法會被淘汰掉
    for i in range(0 , POP_SIZE):
        for j in range(0 , DNA_SIZE):
            if 4 < location_x[i , j] < 6 and 2 < location_y[i , j] < 8:
                fitness[i] = 1e-6

    fitness = fitness / fitness.sum() # 做normalization，確保fitness為1個機率分布
    return fitness , distance_goal


# 根據適應度(機率分布)決定index
# 假如fitness中的[3 10 15 20]比較大
# 那pop中的第3、10、15、20的人的DNA的組合非常有可能被重複抽取出來
# 也就是將適者留下，不適者淘汰的意思
def select(pop_x , pop_y , fitness):
    index = np.random.choice(np.arange(POP_SIZE) , 
                             size = POP_SIZE ,
                             replace = True ,  # replace = True => 取後不放回代表index會重複選取，index會有重複的數字
                             p = fitness)
    return pop_x[index] , pop_y[index]


def crossover(father , mother_candidate):
    crossover_decision = np.random.rand()
    if crossover_decision < CROSS_RATE:
        # 先讓child繼承father全部的基因
        child = father.copy()

        # 決定pop中哪個人要與father交換基因當作mother
        crossover_index = np.random.randint(0 , POP_SIZE , size = 1)
        mother = mother_candidate[crossover_index , :].reshape([-1 , ])

        # 並決定mother的哪幾條DNA要進行交換(cross_points)
        cross_points = np.random.randint(0 , 2 , 2 * DNA_SIZE).astype(np.bool)

        # 將child指定的基因位置換成mother指定的基因位置的基因
        child[cross_points] = mother[cross_points]

        return child

    if crossover_decision >= CROSS_RATE:
        return father

# child中每個element都是移動量
# 而在這個步驟中就是換掉child部分的移動量
def mutate(child):
    for point in range(0 , DNA_SIZE):
        if np.random.rand() < MUTATE_RATE:
            temp = (np.random.randint(INCREMENT_BOUND[0] , INCREMENT_BOUND [1] , size = 1)[0] - 0.5) * RESOLUTION
            child[point] = temp
    return child
#--------------------------主要函數--------------------------#

class Line(object):
    def __init__(self , goal_point , start_point , obstacle_line):
        self.goal_point = goal_point
        self.start_point = start_point
        self.obstacle_line = obstacle_line

        plt.figure(figsize = (20 , 10))
        plt.ion()

    def plotting(self , location_x , location_y):
        plt.cla()
        plt.scatter(*self.start_point, s=200, c='green' , label = 'Start')
        plt.scatter(*self.goal_point, s=200, c='red' , label = 'Goal')
        plt.plot(self.obstacle_line[:, 0], self.obstacle_line[:, 1], lw=6, c='blue' , label = 'wall')
        plt.plot(location_x.T, location_y.T, c='k')
        plt.legend(fontsize = 20)
        plt.xlim([-5 , 15])
        plt.ylim([-5 , 15])
        plt.pause(0.01)


# pop是N X M 的矩陣，N代表有幾種走法(人口)，M代表有幾個城市(DNA)，其中pop[i , :]的每個element都代表一個城市的編號，不會重複
pop_x = np.random.randint(INCREMENT_BOUND[0] , INCREMENT_BOUND [1] , size = [POP_SIZE , DNA_SIZE])
pop_y = np.random.randint(INCREMENT_BOUND[0] , INCREMENT_BOUND [1] , size = [POP_SIZE , DNA_SIZE])

env = Line(GOAL_POINT , START_POINT , OBSTACLE_LINE)

for generation in range(0 , N_GENERATIONS):
    location_x , location_y = translateDNA(pop_x , pop_y)
    fitness , distance_goal = get_fitness(location_x , location_y)

    pop_x , pop_y = select(pop_x , pop_y , fitness)  # 盡可能把一些fitness較低pop[i , :]的剔除，進而留下一些fitness較高的pop[i , :]，將適者留下不適者淘汰的意思
    pop = np.hstack([pop_x , pop_y])
    mother_candidate = pop.copy()
    for i in range(0 , len(pop)):
        father = pop[i , :]
        child = crossover(father , mother_candidate)
        child = mutate(child)
        pop[i , :] = child

    pop_x = pop[: , :DNA_SIZE]
    pop_y = pop[: , DNA_SIZE:]

    best_index = np.argmax(fitness)
    print('Generation : {} , 最短距離 : {:.2f}'.format(generation , distance_goal[best_index]))
    env.plotting(location_x , location_y)
    best_x = location_x[best_index]
    best_y = location_y[best_index]

plt.ioff()
plt.show()





