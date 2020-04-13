import numpy as np
import matplotlib.pyplot as plt

# 用一句話來概括這個算法就是:從一群人中挑出兩個人互相比較，分別為child與father
# 完全不改變father基因，但child必須學習father的一些優點，接著再把兩個人放回人群

DNA_SIZE = 10            # DNA length
POP_SIZE = 20            # population size
CROSS_RATE = 0.6         # mating probability (DNA crossover)
MUTATION_RATE = 0.01     # mutation probability
N_GENERATIONS = 200
X_BOUND = [0 , 5]         # x upper and lower bounds
X_UPPER_BOUND = 5
X_LOWER_BOUND = 0

#--------------------------主要函數--------------------------#
# 想要找F(x) = np.sin(10 * x) * x + np.cos(2 * x) * x 的最大值
def F(x):
    return np.sin(10 * x) * x + np.cos(2 * x) * x


def translateDNA(pop):
    binary_to_decimal = np.dot(pop , 2 ** np.arange(DNA_SIZE)[::-1])
    normalization = binary_to_decimal / float(2**DNA_SIZE - 1) * X_UPPER_BOUND
    return normalization


def get_fitness(product):
    return product


def learn_from_father(child_father):
    # 決定father有幾條DNA去替代child相對應的基因
    crossover_num = np.random.choice(np.arange(DNA_SIZE) , size = 1) + 1

    # 決定哪幾條DNA去替代child相對應的基因
    cross_points = np.random.choice(np.arange(DNA_SIZE) , size = crossover_num)

    # 將child指定的基因位置換成father指定的基因位置的基因
    child_father[0 , cross_points] = child_father[1 ,cross_points]
    return child_father


# 若要變種的話，就是讓某些基因1變0、0變1
def mutate(child_father):
    for point in range(0 , DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            if child_father[0 , point] == 0:
                child_father[0 , point] = 1
            elif child_father[0 , point] == 1:
                child_father[0 , point] = 0
    return child_father
#--------------------------主要函數--------------------------#



# 隨機決定人口矩陣
pop = np.random.randint(0 , 2 , size = (POP_SIZE , DNA_SIZE)).astype(np.int8)

x = np.linspace(X_LOWER_BOUND , X_UPPER_BOUND , 200)
plt.ion()
plt.figure(figsize = (20 , 10))
plt.plot(x , F(x))

for generation in range(0 , N_GENERATIONS):
    for _ in range(0 , 10):  # random pick and compare n times
        pick_pop_index = np.random.choice(np.arange(0 , POP_SIZE) , size = 2 , replace = False)
        pick_pop = pop[pick_pop_index]                 # 隨機從pop選出2個人(child與father)並將兩者合併當作pick_pop
        product = F(translateDNA(pick_pop))
        fitness = get_fitness(product)                 # 計算pick_pop中2個人各自的適應度
        child_father_index = np.argsort(fitness)       # child_father_index 為兩者適應度大小的排序
        child_father = pick_pop[child_father_index]    # 讓pick_pop按照child_father_index來排列，child是第0個，father是第1個
        child_father = learn_from_father(child_father) # 隨機決定father中某幾條基因去替代child相對應的基因
        child_father = mutate(child_father)
        pop[pick_pop_index] = child_father

    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(translateDNA(pop) , F(translateDNA(pop)) , s = 200 , lw = 0 , c = 'red' , alpha = 0.5)
    plt.pause(0.05)

    best_pop = pop[np.argmax(fitness) , :] # 挑出最好的人口(挑出fitness最大的index)
    print('Generation : {} , 最大值 : {:.2f}'.format(generation , F(translateDNA(best_pop)) ) )

plt.ioff()
plt.show()
