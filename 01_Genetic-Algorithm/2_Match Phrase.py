import numpy as np

# 要從ASCII_CODE的搜索範圍中找出與TARGET_PHRASE最相符的字符
TARGET_PHRASE = 'Fuck you! Ass hole!'

DNA_SIZE = len(TARGET_PHRASE)       # DNA的長度
POP_SIZE = 300                      # 總人口數
CROSS_RATE = 0.4                    # 交配的機率
MUTATION_RATE = 0.01                # 變種的機率
N_GENERATIONS = 1000                # 共迭代1000次

TARGET_ASCII = np.fromstring(TARGET_PHRASE , dtype = np.uint8)  # 將字串換成數字
ASCII_BOUND = [32 , 127] # ASCII_CODE的搜索範圍

#--------------------------主要函數--------------------------#
# 將數字轉回字串
def translateDNA(DNA):
    return DNA.tostring().decode('ascii')


# pop[20] = [1 2 3 1 10 9] <-> TARGET_ASCII = [1 1 3 9 10 8]
# 只有第1、3、5，這幾個index有對應到TARGET_ASCII的內容，所以fitness[20]
# 而fitness[i]當然就是越大越有機會讓pop[i]被挑出來
def get_fitness(pop):
    # temp = (pop == TARGET_ASCII)
    # fitness = np.zeros([300 , ])
    # for i in range(0 , temp.shape[0]):
    #     count = 0
    #     等於3for j in range(0 , temp.shape[1]):
    #         if temp[i , j] == True:
    #             count += 1
    #     fitness[i] = count
    # fitness = fitness.astype(np.float16)
    # fitness = fitness - np.min(fitness)
    # fitness += 1e-3
    # fitness = fitness / fitness.sum()

    # 假如fitness => [0 , 0 , 0.25 , 0.25 , 0.25 , 0.25]
    # 則"第0"與"第1"這兩個index根本無法被sample到
    # 所以讓fitness加上1e-3，可以確保"第0"與"第1"這兩個index有機會被被sample到
    fitness = (pop == TARGET_ASCII).sum(axis = 1).astype(np.float16)
    fitness = fitness - np.min(fitness)
    fitness += 1e-3
    fitness = fitness / fitness.sum()
    return fitness


# 根據適應度(機率分布)決定index
# 假如fitness中的[3 10 15 20]比較大
# 那pop中的第3、10、15、20的人的DNA的組合非常有可能被重複抽取出來
# 也就是將適者留下，不適者淘汰的意思
def select(pop , fitness):
    index = np.random.choice(np.arange(POP_SIZE) ,
                             size = POP_SIZE ,
                             replace = True , 
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


# 若要變種的話，就是在ASCII_BOUND這個區間選一個值去替代child[point]
def mutate(child):
    for point in range(0 , DNA_SIZE):
            if np.random.rand() < MUTATION_RATE:
                child[point] = np.random.randint(ASCII_BOUND[0] , ASCII_BOUND[1])
    return child
#--------------------------主要函數--------------------------#

# 隨機決定人口矩陣
pop = np.random.randint(ASCII_BOUND[0] , ASCII_BOUND[1] , size = (POP_SIZE , DNA_SIZE)).astype(np.int8)

for generation in range(0 , N_GENERATIONS):
    fitness = get_fitness(pop)    # step1 算出所有人口中每個人的適應度，並將fitness轉換成機率分布，作為sample的依據
    pop = select(pop , fitness)   # step2 盡可能把一些fitness較低pop[i , :]的剔除，進而留下一些fitness較高的pop[i , :]，將適者留下不適者淘汰的意思

    mother_candidate = pop.copy()
    for i in range(0 , len(pop)):
        father = pop[i , :]
        child = crossover(father , mother_candidate)   # 開始將適者與pop中某個人一部分的基因互相交換
        child = mutate(child)                          # 將child中的某些基因換掉
        pop[i , :] = child                             # 以child將pop[i , :]替代掉

    best_DNA = pop[np.argmax(fitness)] # 挑出最好的人口(挑出fitness最大的index)
    best_phrase = translateDNA(best_DNA)
    print('Generation : {} , Phrase : {}'.format(generation , best_phrase))

    if best_phrase == TARGET_PHRASE:
        break
