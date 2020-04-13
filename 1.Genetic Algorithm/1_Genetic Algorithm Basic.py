import numpy as np
import matplotlib.pyplot as plt

'''
基因演算法概念:
先隨機產生出N組DNA(0與1組合之M維向量)，堆疊成人口矩陣(N X M之矩陣代表有N個人，每個人有M個DNA)
再計算每組DNA的適應程度並轉換為機率分布，接著再以此機率分布挑從人口矩陣選出中N個適應度較高的人(會有重複的人出現)，也藉此淘汰掉適應度較低的人
再讓人口矩陣裡的每個人隨機互相交配，就是交換DNA產生後代
並讓某些後代的基因產生變異(1變0、0變1)，再以此後代更新人口矩陣
最後在讓此過程不斷循環 
'''

DNA_SIZE = 10            # DNA的長度
POP_SIZE = 100           # 總人口數
CROSS_RATE = 0.8         # 交配的機率
MUTATION_RATE = 0.003    # 變種的機率
N_GENERATIONS = 200      # 共迭代200次
X_UPPER_BOUND = 5        # x upper bounds
X_LOWER_BOUND = 0        # x lower bounds


#--------------------------主要函數--------------------------#
# 想要找F(x) = np.sin(10 * x) * x + np.cos(2 * x) * x 的最大值
def F(x): 
    return np.sin(10 * x) * x + np.cos(2 * x) * x     


# 得到適應度，而適應度其實就是機率分布
def get_fitness(fitness):
    fitness = fitness - np.min(fitness)   # 讓所有prob[i]大於等於零
    # 假如fitness => [0 , 0 , 0.25 , 0.25 , 0.25 , 0.25]
    # 則"第0"與"第1"這兩個index根本無法被sample到
    # 所以讓fitness加上1e-3，可以確保"第0"與"第1"這兩個index有機會被sample到
    fitness += 1e-3
    fitness = fitness / fitness.sum() # 做normalization，確保fitness為1個機率分布
    return fitness


# 將2進位轉為10進位
# (pop)_2x10 = [1 1 1 1 1 1 1 1 1
#               1 0 0 0 0 0 0 0 0]#
# (binary)_10x1 = [512 256 128 64 32 16 8 4 2 1].T
# (pop * binary)_2x1 = [1023 512].T
def translateDNA(pop):
    binary = 2 ** np.arange(DNA_SIZE)[::-1]
    binary_to_decimal = np.dot(pop , binary)
    normalization = binary_to_decimal / float(2**DNA_SIZE - 1) * (X_UPPER_BOUND - X_LOWER_BOUND)
    return normalization


# 根據適應度(機率分布)決定index
# 假如fitness中的[3 10 15 20]比較大
# 那pop中的第3、10、15、20的人的DNA的組合非常有可能被重複抽取出來
# 也就是將適者留下，不適者淘汰的意思
def select(pop , fitness):
    index = np.random.choice(np.arange(POP_SIZE) , 
                             size = POP_SIZE ,
                             replace = True ,  # replace = True => 取後不放回代表index會重複選取，index會有重複的數字
                             p = fitness)
    return pop[index]


# 開始將father與mother一部分的基因互相交換
def crossover(father , mother_candidate):
    crossover_decision = np.random.rand()
    if crossover_decision < CROSS_RATE:
        # 先讓child繼承father全部的基因
        child = father.copy()

        # 決定pop中哪個人要與father交換基因當作mother
        crossover_index = np.random.randint(0 , POP_SIZE , size = 1)
        mother = mother_candidate[crossover_index , :].reshape([-1 , ])

        # 決定mother有幾條DNA要進行交換(crossover_num)
        crossover_num = np.random.choice(np.arange(DNA_SIZE) , size = 1) + 1

        # 並決定mother的哪幾條DNA要進行交換(cross_points)
        cross_points = np.random.choice(np.arange(DNA_SIZE) , size = crossover_num[0] , replace = False)

        # 將child指定的基因位置換成mother指定的基因位置的基因
        child[cross_points] = mother[cross_points]

        return child

    if crossover_decision >= CROSS_RATE:
        return father


# 若要變種的話，就是讓某些基因1變0、0變1
def mutate(child):
    for point in range(0 , DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            if child[point] == 0:
                child[point] = 1
            elif child[point] == 1:
                child[point] = 0
    return child
#--------------------------主要函數--------------------------#


# 隨機決定人口矩陣
pop = np.random.randint(2 , size = (POP_SIZE , DNA_SIZE))

x = np.linspace(X_LOWER_BOUND , X_UPPER_BOUND , 200)
plt.ion()       # something about plotting
plt.figure(figsize = (20 , 10))
plt.plot(x , F(x))

for generation in range(0 , N_GENERATIONS):
    F_values = F(translateDNA(pop))    # compute function value by extracting DNA

    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(translateDNA(pop) , F_values , s = 200, lw = 0 , c = 'red' , alpha = 0.5)
    plt.pause(0.1)

    fitness = get_fitness(F_values) # 算出所有人口中每個人的適應度，並將fitness轉換成機率分布，作為sample的依據
    pop = select(pop , fitness)     # 盡可能把一些fitness較低pop[i , :]的剔除，進而留下一些fitness較高的pop[i , :]，將適者留下不適者淘汰的意思
            
    mother_candidate = pop.copy()
    for i in range(0 , len(pop)):
        father = pop[i , :]
        child = crossover(father , mother_candidate)   # 開始將適者與pop中某個人一部分的基因互相交換
        child = mutate(child)                          # 將child中的某些基因1變0、0變1
        pop[i , :] = child                             # 以child將pop[i , :]替代掉

    best_pop = pop[np.argmax(fitness) , :] # 挑出最好的人口(挑出fitness最大的index)
    print('Generation : {} , 適應力最好的人口基因 : {}'.format(generation , best_pop))

plt.ioff()
plt.show()