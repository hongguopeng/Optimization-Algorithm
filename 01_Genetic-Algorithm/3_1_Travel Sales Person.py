import matplotlib.pyplot as plt
import numpy as np

DNA_SIZE = 20         # DNA的長度(代表有20座城市)
POP_SIZE = 500        # 總人口數(代表有500種走法)
CROSS_RATE = 0.1      # 交配的機率
MUTATE_RATE = 0.02    # 變種的機率
N_GENERATIONS = 500   # 共迭代200次


#--------------------------主要函數--------------------------#
# location_x => 記錄每種走法，所經過的城市的x的座標
# location_y => 記錄每種走法，所經過的城市的y的座標
def translateDNA(pop , city_position):     # city_position 代表每一個城市的座標(x , y)
    location_x = np.zeros([pop.shape[0] , pop.shape[1]])
    location_y = np.zeros([pop.shape[0] , pop.shape[1]])
    for i in range(0 , pop.shape[0]):
        for j in range(0 , pop.shape[1]):
            city_index = pop[i , j] # 第i種走法，經過第j個city
            location_x[i , j] = city_position[ city_index , 0 ]  # 若是pop[2 , 3]為城市16，則location_x[2 , 3]就是城市16的x座標
            location_y[i , j] = city_position[ city_index , 1 ]  # 若是pop[2 , 3]為城市16，則location_y[2 , 3]就是城市16的y座標
    return location_x , location_y


# 得到適應度，而適應度其實就是機率分布
# 而這裡的適應度其實就是經過每個城市所經過的總距離(total_distance)，再經過np.exp(DNA_SIZE * 2 / total_distance)
# total_distance越大，fitness越低
def get_fitness(location_x , location_y):
    total_distance = np.zeros([location_x.shape[0], ])
    for i in range(0 , location_x.shape[0]):
        for j in range(0 , location_x.shape[1] - 1):
            x_diff = location_x[i , j + 1] - location_x[i , j] # 兩座城市間x座標的差值
            y_diff = location_y[i , j + 1] - location_y[i , j] # 兩座城市間y座標的差值
            distance = (x_diff ** 2 +  y_diff ** 2) ** 0.5 # 兩座城市間的距離
            total_distance[i] = total_distance[i] + distance  # 計算按照pop[i , :]的城市編號走過所有城市的總距離
    fitness = np.exp(DNA_SIZE * 2 / total_distance) # 走過的距離越長，fitness越低。若是距離差一點點，也會因為np.exp()加大差距
    fitness = fitness + 1e-3
    fitness = fitness / fitness.sum()
    return fitness , total_distance


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


"""
father = [0 , 1 , 2 , 3]
mother = [3 , 2 , 1 , 0]
cross_points = [m , f , m , f] (交叉點 => m : 媽媽  f : 爸爸)
child = [3 , 1 , 1 , 3]
那麼這樣的child要經過兩次城市3，兩次城市1，而沒有經過2，
顯然不合理，所以crossover以及mutation都要換一種方式進行

father = [0 , 1 , 2 , 3]
cross_points = [_ , f , _ , f]
child = [3 , 1 , _ , _] (先將爸爸的點填到孩子的前面)
此時除開來自爸爸的1與3，還有0, 2兩個城市，
但是0與2的順序就按照媽媽DNA的先後順序排列，剩下的就取出mother = [3 , 2 , 1 , 0]的0與2兩座城市
接著按照這個順序補去孩子的DNA
child = [1 , 3 , 2 , 0]
"""
def crossover(father , mother_candidate):
    crossover_decision = np.random.rand()
    if crossover_decision < CROSS_RATE:
        # 先隨機從全部的走法中隨機選出要與father交配的mother走法
        crossover_index = np.random.randint(0 , POP_SIZE , size = 1)
        mother = mother_candidate[crossover_index[0] , :]

        # 隨機決定father這個走法中有哪些城市是要被取出來的，把這些城市抓出來當作精子
        crossover_num = np.random.choice(np.arange(DNA_SIZE) , size = 1) + 1
        cross_points = np.random.choice(np.arange(DNA_SIZE) , size = crossover_num[0] , replace = False)
        sperm_city = father[cross_points]
        
        # 取出mother這個走法所經過的城市，但排除已經在sperm_city中已經的城市即為卵子
        # 也就是ovum_city就是要從mother中挑出sperm_city所沒有的城市
        ovum_city = [city for city in mother if city not in sperm_city]
        ovum_city = np.array(ovum_city)
#        count = 0
#        ovum_city = np.zeros([DNA_SIZE - sperm_city.shape[0] , ]).astype(np.int8)
#        for city in mother:
#            if city not in sperm_city:
#                ovum_city[count] = city
#                count += 1
        child = np.concatenate([sperm_city , ovum_city])
        return child
    elif crossover_decision >= CROSS_RATE:
        return father


# 在child走過的城市只能互相交換，才不會有重複的城市
def mutate(child):
    for point in range(0 , DNA_SIZE):
        if np.random.rand() < MUTATE_RATE:
            mutate_point = np.random.randint(0 , DNA_SIZE)
            temp = child[point]
            child[point] = child[mutate_point]
            child[mutate_point] = temp
    return child
#--------------------------主要函數--------------------------#

class TravelSalesPerson(object):
    def __init__(self , DNA_SIZE):
        self.city_position = np.random.rand(DNA_SIZE , 2) # 隨機產生城市的座標
        plt.figure(figsize = (20 , 10))
        plt.ion()

    def plotting(self , lx , ly , total_d):
        plt.cla()
        plt.scatter(self.city_position[: , 0].T , self.city_position[: , 1].T , s = 100 , c = 'red')
        plt.plot(lx.T , ly.T , 'g-')
        plt.text(-0.05 , -0.05 , 'Total distance = {:.2f}'.format(total_d) , fontdict={'size': 20, 'color': 'blue'})
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.pause(0.01)


# pop是N X M 的矩陣，N代表有幾種走法(人口)，M代表有幾個城市(DNA)，其中pop[i , :]的每個element都代表一個城市的編號，不會重複
pop = []
for _ in range(0 , POP_SIZE): 
    pop.append(np.random.permutation(DNA_SIZE))
pop = np.array(pop)


env = TravelSalesPerson(DNA_SIZE)
for generation in range(0 , N_GENERATIONS):
    lx , ly = translateDNA(pop , env.city_position)
    fitness , total_distance = get_fitness(lx , ly)

    pop = select(pop , fitness)  # 盡可能把一些fitness較低pop[i , :]的剔除，進而留下一些fitness較高的pop[i , :]，將適者留下不適者淘汰的意思
    mother_candidate = pop.copy()
    for i in range(0 , len(pop)):
        father = pop[i , :]
        child = crossover(father , mother_candidate)
        child = mutate(child)
        pop[i , :] = child

    best_index = np.argmax(fitness)
    print('Generation : {} , 最短距離 : {:.2f}'.format(generation , total_distance[best_index]))
    env.plotting(lx[best_index] , ly[best_index] , total_distance[best_index]) # 把每次最好的走法拿去畫圖

plt.ioff()
plt.show()