import numpy as np
import time
import matplotlib.pyplot as plt

'''
ES DNA 形式分兩種，第1個DNA是控制數值，第2個DNA是控制這個數值的變異強度
比如一個問題有4個變數
那第一個DNA中就有4個位置存放這4個變數的值(這就是我們要得到的答案)
第2個DNA中就存放4個變數的變動幅度值

例如:
DNA1 = 1.23, -0.13, 2.35, 112.5 可以看成為4個Normal分布的4個平均值
DNA2 = 0.1, 2.44, 5.112, 2.144  可以看成為4個Normal分布的4個標準差
所以這2條DNA 都需要被crossover和mutate

傳統的基因演算法只能處理離散值，而這種演算法可以更加輕鬆自在的在實數區間上進行變異
'''

DNA_SIZE = 1             # DNA的長度
POP_SIZE = 100           # 總人口數
N_GENERATIONS = 200      # 共迭代200次
N_child = 50               # 生出50個孩子
X_UPPER_BOUND = 5        # x upper bounds
X_LOWER_BOUND = 0        # x lower bounds


#--------------------------主要函數--------------------------#
def F(x):
    return np.sin(10 * x) * x + np.cos(2 * x) * x     # to find the maximum of this function


# find non-zero fitness for selection
def get_fitness(pred): 
    return pred.flatten()


def deliver_child(pop , n_child):
    childs = {'DNA' : np.zeros([n_child , DNA_SIZE]) , 'mutate_strength' : np.zeros([n_child , DNA_SIZE])}

    # 決定childs中每個人的DNA與mutate_strength
    for i in range(0 , childs['DNA'].shape[0]):
        index = np.random.choice(np.arange(POP_SIZE) , size = 2 , replace = False)
        father_index = index[0]
        mother_index = index[0]
    
        # 決定第i個child到底要繼承father或mother的DNA與mutate_strength
        # decision_from_father_mother => 繼承father的DNA與mutate_strength
        # decision_from_father_mother => 繼承mother的DNA與mutate_strength
        decision_from_father_mother = np.random.randint(0 , 2)
        if decision_from_father_mother == 0:
            inherit_mean = pop['DNA'][father_index]
            inherit_var  = pop['mutate_strength'][father_index]
        elif decision_from_father_mother == 1:
            inherit_mean = pop['DNA'][mother_index]
            inherit_var  = pop['mutate_strength'][mother_index]
    
        # mutate_strength也能變異，將mutate_strength變異以後，就能在快收斂的時候自覺的逐漸減小變異強度，方便收斂
        decision_mutate = np.random.randint(0 , 2)
        if decision_mutate == 0:
            inherit_var = np.maximum(inherit_var + np.random.rand() , 0.) # 可以保證inherit_var一定會大於0
        elif decision_mutate == 1:
            inherit_var = np.maximum(inherit_var - np.random.rand() , 0.) # 可以保證inherit_var一定會大於0
    
        # 繼承father或mother的DNA後，稍微做一下變異
        # 可以想成是在(inherit_mean , inherit_mean)的normal dist上隨機取一個點當作變異後的inherit_mean
        inherit_mean = inherit_mean + inherit_var * np.random.randn()

        # 但變異後的inherit_mean可能會超出[X_LOWER_BOUND , X_UPPER_BOUND]這個範圍，所以在這裡做一個clip
        inherit_mean = np.clip(inherit_mean , X_LOWER_BOUND , X_UPPER_BOUND)

        childs['DNA'][i] = inherit_mean
        childs['mutate_strength'][i] = inherit_var
    return childs

# 將pop與childs放在一塊來比較適應度，留下適應度好的，並將適應度差的淘汰
def select(pop , childs):
    # 將pop與childs放在一塊變成新的pop
    for key in ['DNA' , 'mutate_strength']:
        pop[key] = np.concatenate([pop[key] , childs[key]] , axis = 0)

    # 計算pop中所有人的適應度，並且由大到小做排名，取前POP_SIZE名適應度好的人
    fitness = get_fitness(F(pop['DNA']))
    survive_index = fitness.argsort()[::-1][:POP_SIZE]

    for key in ['DNA' , 'mutate_strength']:
        pop[key] = pop[key][survive_index]

    return pop
#--------------------------主要函數--------------------------#


# 隨機生成每個人口的DNA與mutate_strength
pop = {'DNA' : X_UPPER_BOUND * np.random.rand(POP_SIZE , DNA_SIZE) ,
       'mutate_strength' : np.random.rand(POP_SIZE , DNA_SIZE)}


x = np.linspace(X_LOWER_BOUND , X_UPPER_BOUND , 200)
plt.ion()
plt.figure(figsize = (20 , 10))
plt.plot(x , F(x))

for generation in range(0 , N_GENERATIONS):

    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(pop['DNA'] , F(pop['DNA']) , s = 200, lw = 0 , c = 'red' , alpha = 0.5)
    plt.pause(0.5)

    childs = deliver_child(pop , N_child)
    pop = select(pop , childs)

    best_DNA = pop['DNA'].max()
    best_mutate_strength = pop['mutate_strength'].max()
    print('Generation : {} , DNA : {:.2f} , mutate_strength : {:.2f}'.format(generation , best_DNA , best_mutate_strength))
    
plt.ioff()
plt.show()

