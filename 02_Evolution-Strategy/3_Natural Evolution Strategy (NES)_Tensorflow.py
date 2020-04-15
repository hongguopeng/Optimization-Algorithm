import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance

DNA_SIZE = 2         # 代表x1座標與x2座標
POP_SIZE = 20        # 總人口數
N_GENERATION = 200   # 共迭代200次
LR = 0.02            # learning rate


# fitness function : 找-(x_1^2 + x_2^2)的最大值
def get_fitness(pred):
    x_1 = pred[: , 0]
    x_2 = pred[: , 1]
    return -(x_1 ** 2 + x_2 ** 2)
    # return -(x_1**2 + 2*x_2**2 - 0.3*np.cos(3*np.pi*x_1) - 0.4*np.cos(4*np.pi*x_2) + 0.7)


# 畫等高線圖
def contour():
    plt.figure(figsize = (20 , 10))
    n = 300
    x = np.linspace(-20 , 20 , n)
    X , Y = np.meshgrid(x , x)
    Z = np.zeros_like(X)
    for i in range(0 , n):
        for j in range(0 , n):
            Z[i , j] = get_fitness(np.array([[x[i] , x[j]]]))
    plt.contourf(X , Y , Z , 100 , alpha = 0.75 , cmap = plt.cm.rainbow)
    plt.ylim(-20 , 20)
    plt.xlim(-20 , 20)
    plt.ion()


#--------------------------建立神經網路--------------------------#
''' 
build multivariate distribution(雙變數的normal distribution)
現在有 X1與X2兩個r.v
初始 cov 為 tf.eye(DNA_SIZE = 2) => 對角矩陣，代表X1與X2兩個r.v為不相關(不是獨立)!!
normal_dist.sample(POP_SIZE=20) = [x1_1    x1_2
                                   x2_1    x2_2
                                   ...   ...
                                   x20_1   x20_2] 
normal_dist.sample(POP_SIZE=20)代表讓X1與X2依照定義的 mean 與 cov 所形成的 distribution 以 sample 出20個值，並把這20個值當作訓練的數據丟進神經網路
而loss所要微分的對象即是 mean 與 var
'''
mean = tf.Variable(tf.random_normal([DNA_SIZE , ] , 5. , 1.) , dtype = tf.float32 , name = 'mean')
cov = tf.Variable(3. * tf.eye(DNA_SIZE) , dtype = tf.float32 , name = 'cov')
normal_dist = MultivariateNormalFullCovariance(loc = mean , covariance_matrix = cov)
make_child = normal_dist.sample(POP_SIZE) # 在定義好的normal_dist上sample資料

childs_fitness_input = tf.placeholder(tf.float32 , [POP_SIZE , ])
childs_input = tf.placeholder(tf.float32 , [POP_SIZE , DNA_SIZE])
loss = -tf.reduce_mean(normal_dist.log_prob(childs_input) * childs_fitness_input) # log prob * fitness
train_op = tf.train.GradientDescentOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#--------------------------建立神經網路--------------------------#


# 畫等高線圖
contour()

# 開始訓練神經網路
for generation in range(0 , N_GENERATION):

    childs = sess.run(make_child)
    childs_fitness = get_fitness(childs)
    sess.run(train_op , feed_dict = {childs_fitness_input : childs_fitness , childs_input : childs})

    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(childs[: , 0] , childs[: , 1] , s = 30 , c = 'white')
    plt.pause(0.1)

    print('Generation : {} , 最大值 : {:.2f}'.format(generation , childs_fitness.max()))

plt.ioff()
plt.show()
