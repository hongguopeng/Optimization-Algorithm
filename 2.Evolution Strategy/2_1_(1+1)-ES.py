import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 1             # DNA的長度
N_GENERATIONS = 200      # 共迭代200次
mutate_strength = 10.    # 初始變異強度
X_UPPER_BOUND = 5        # x upper bounds
X_LOWER_BOUND = 0        # x lower bounds


#--------------------------主要函數--------------------------#
def F(x):
    return np.sin(10 * x) * x + np.cos(2 * x) * x     # to find the maximum of this function


# find non-zero fitness for selection
def get_fitness(pred): 
    return pred.flatten()


def mutate(father , mutate_strength):

    # child是由father變異生成的，而變異的方式就像是在normal distribution隨機sample一個點
    child = father + mutate_strength * np.random.randn(DNA_SIZE)

    # 但變異後的child可能會超出[X_LOWER_BOUND , X_UPPER_BOUND]這個範圍，所以在這裡做一個clip
    child = np.clip(child , X_LOWER_BOUND , X_UPPER_BOUND)

    return child


def select(father , child , mutate_strength):
    fitness_father = get_fitness(F(father))[0]
    fitness_child = get_fitness(F(child))[0]

    # child比father適應度還要高，代表還沒"快要收斂"
    # 增加mutate_strength以擴大搜索範圍
    if fitness_father <= fitness_child:
        father = child
        ps = 1.
    # child比father適應度還要低，代表"快要收斂"
    # 減少mutate_strength以縮小搜索範圍
    elif fitness_father > fitness_child:
        ps = 0.

    # ps = 1. => np.exp(1 / np.sqrt(DNA_SIZE + 1)  * (ps - p_target)/(1 - p_target)) ~= 1.4
    # 增加mutate_strength以擴大搜索範圍
    # ps = 1. => np.exp(1 / np.sqrt(DNA_SIZE + 1)  * (ps - p_target)/(1 - p_target)) ~= 0.92
    # 減少mutate_strength以縮小搜索範圍
    p_target = 1/10 # 將p_target調小，更有利於擴大搜索範圍
    mutate_strength *= np.exp(1 / np.sqrt(DNA_SIZE + 1)  * (ps - p_target)/(1 - p_target))
    return father , mutate_strength
#--------------------------主要函數--------------------------#


x = np.linspace(X_LOWER_BOUND , X_UPPER_BOUND , 200)
plt.ion()
plt.figure(figsize = (20 , 10))
plt.plot(x , F(x))

# 隨機生成father的DNA
father = X_UPPER_BOUND * np.random.rand(DNA_SIZE)

for generation in range(0 , N_GENERATIONS):

    # 將father的DNA依據mutate_strength去變異，產生child
    child = mutate(father , mutate_strength)

    # 比較father與child的適應度，決定誰該生存下來，並改變mutate_strength
    father , mutate_strength = select(father , child , mutate_strength)
    father_y = F(father)
    child_y = F(child)

    if 'sca_scatter_1' in globals():
        sca_scatter_1.remove()
        sca_scatter_2.remove()
        sca_legend.remove()

    sca_scatter_1 = plt.scatter(father , father_y , s = 200 , lw = 0 , c = 'red' , alpha = 0.5 , label = 'father')
    sca_scatter_2 = plt.scatter(child , child_y , s = 200 , lw = 0 , c = 'blue' , alpha = 0.5 , label = 'child')
    sca_legend = plt.legend(fontsize = 20)
    plt.pause(0.001)

    best_DNA = father[0]
    print('Generation : {} , DNA : {:.2f} , mutate_strength : {:.2f}'.format(generation , best_DNA , mutate_strength))


plt.ioff()
plt.show()
