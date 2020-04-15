import numpy as np
# 在某些電腦無法在cmd中找不到gym這個模組，所以加上下面這兩行，就可以在cmd中執行了
import sys
sys.path.append('C:\\Users\hong guo peng\\Anaconda3\\Lib\\site-packages') # gym模組所在的位置
import gym 
import multiprocessing as mp

N_SCOUT = 10                 # half of the training population
N_GENERATION = 5000          # training step
LR = 0.05                    # learning rate
SIGMA = 0.05                 # mutation strength or step size
N_CORE = mp.cpu_count() - 1
CONFIG = [dict(game = 'CartPole-v0' ,
               n_feature = 4 , 
               n_action = 2 ,
               continuous_a = [False] ,
               ep_max_step = 700 ,
               eval_threshold = 650) ,
          
          dict(game = 'MountainCar-v0' ,
               n_feature = 2 ,
               n_action = 3 ,
               continuous_a = [False] ,
               ep_max_step = 200 , 
               eval_threshold = -120) ,

          dict(game = 'Pendulum-v0' ,
               n_feature = 3 ,
               n_action = 1 ,
               continuous_a = [True , 2.] , 
               ep_max_step = 200 ,
               eval_threshold = -200)    ]
CONFIG = CONFIG[0]  # 選1個game來玩

# 將1維陣列的params，reshape成matrix的樣子
def params_reshape(shapes , params):
    weight_list = []
    bias_list = []
    for i in range(0 , len(shapes)):
        weight_list.append(shapes[i])
        bias_list.append(shapes[i][1])

    reshaped_params = []
    for i in range(0 , len(weight_list)):
        index_for_weight = weight_list[i][0] * weight_list[i][1]
        index_for_bias = weight_list[i][1]
    
        weight_temp = params[0 : index_for_weight].reshape(weight_list[i][0] , weight_list[i][1])
        params = np.delete(params , range(0 , index_for_weight))
    
        bias_temp = params[0 : index_for_bias]
        params = np.delete(params , range(0 , index_for_bias))
    
        reshaped_params.extend([weight_temp , bias_temp])
         
    return reshaped_params


def get_reward(shapes , params , env , ep_max_step , continuous_a , seed_and_id = None):
    if seed_and_id is not None:
        # 利用同一個seed生乘的noise雖然是相同的
        # 但index為奇數所產生的noise與index為偶數所產生的noise，正負號相反
        # 這裡用到的假設 : 這兩種noise其中一個會使output遠離target，那另外一個一定會使output接近target
        [seed , index] = seed_and_id
        np.random.seed(seed)
        if index % 2 == 0:
            sign = 1
        elif index % 2 == 1:
            sign = -1
        noise = np.random.randn(params.shape[0])
        params = params + SIGMA * sign * noise

    reshaped_params = params_reshape(shapes , params)
                    
    # 開始與環境互動
    state = env.reset()
    cum_reward = 0.
    for step in range(0 , ep_max_step):
        action = get_action(reshaped_params , state , continuous_a)
        state , reward , done , _ = env.step(action)
        # mountain car's reward can be tricky
        if env.spec._env_name == 'MountainCar' and state[0] > -0.1:
            reward = 0.
        cum_reward += reward
        if done: break

    if seed_and_id is None:
        return cum_reward
    else:
        return [cum_reward , seed , sign]


def get_action(params , in_ , continuous_a):
    in_ = in_.reshape([1 , len(in_)])          # input layer
    h_1 = np.dot(in_ , params[0]) + params[1]  # hidden layer_1
    h_1 = np.tanh(h_1)                         # 將h_1丟進activaction function=>tanh
    h_2 = np.dot(h_1 , params[2]) + params[3]  # hidden layer_2
    h_2 = np.tanh( h_2)                        # 將h_2丟進activaction function=>ReLU
    out = np.dot(h_2 , params[4]) + params[5]  # output layer
    if not continuous_a[0]: 
        return np.argmax(out , axis = 1)[0]      # discrete action
    else: 
        return continuous_a[1] * np.tanh(out)[0] # continuous action


def build_net():
    def linear(neuron_in , neuron_out):
        weight = np.random.randn(neuron_in * neuron_out).astype(np.float32) * 0.1 # 得到初始weight
        bias = np.random.randn(neuron_out).astype(np.float32) * 0.1        # 得到初始bias
        shapes = [neuron_in , neuron_out]
        params = np.concatenate([weight , bias])
        return shapes , params

    shape_0 , param_0 = linear(CONFIG['n_feature'] , 30) # input param
    shape_1 , param_1 = linear(30 , 20)                  # hidden param
    shape_2 , param_2 = linear(20 , CONFIG['n_action'])  # output param
    shapes_set = [shape_0 , shape_1 , shape_2]
    params_set = np.concatenate([param_0 , param_1 , param_2])
    return shapes_set , params_set , np.zeros_like(params_set).astype(np.float32)


def train(net_shapes , net_params , move , weight , pool):
    # 在這裡做mirrored sampling
    # noise = [1 , 2 , 3] => noise_seed = [1 , 1 , 2 , 2 , 3 , 3]
    seed = np.random.choice(a = N_SCOUT  , size = N_SCOUT , replace = False)
    noise_seed = seed.repeat(2)

    # 以multiprocessing的方式與環境互動並得到reward
    jobs = [pool.apply_async(get_reward , (net_shapes ,
                                           net_params ,
                                           env ,
                                           CONFIG['ep_max_step'] ,
                                           CONFIG['continuous_a'] ,
                                           [noise_seed[scout] , scout] ) )\
            for scout in range(0 , N_SCOUT * 2)]

    rewards , seeds , signs = [] , [] , []
    for job in jobs:
        temp = job.get()
        rewards.append(temp[0])
        seeds.append(temp[1])
        signs.append(temp[2])
    rewards = np.array(rewards)

    rewards_index = np.argsort(rewards)[::-1]           # 得到rewards由大到小的index
    cumulative_update = np.zeros_like(net_params)       # 初始化cumulative_update
    for i , reward_index in enumerate(rewards_index):
        np.random.seed(seeds[reward_index])       # 用seed重建noise
        noise = np.random.randn(net_params.size)  # 用seed重建noise
        sign = signs[reward_index]

        # reward越大的noise[reward_index , :]，佔的權重越大
        cumulative_update = cumulative_update + weight[i] * sign * noise

    # move => 9成的比例遵循原來的方向，另外1成遵循cumulative_update的方向
    momentum = 0.9
    move = momentum * move + (1. - momentum) * cumulative_update/(N_SCOUT * SIGMA)
    update = LR * move
    net_param_update = net_params + update
    
    return net_param_update , move , rewards


if __name__ == "__main__":
    # 並不是直接用reward當權重，而是用事先固定好的weight當作權重
    N_SCOUT_ = N_SCOUT * 2   # 乘2=>是為了mirrored sampling
    rank = np.arange(1 , N_SCOUT_ + 1)
    weight = np.maximum(0 , np.log(N_SCOUT_ / 2 + 1) - np.log(rank))
    weight = weight / weight.sum() - 1 / N_SCOUT_

    # training
    net_shapes , net_params , move = build_net()
    env = gym.make(CONFIG['game']).unwrapped
    pool = mp.Pool(processes = N_CORE)
    moving_average_reward = None      # moving average reward
    for generation in range(0 , N_GENERATION):
        net_params , move , training_reward = train(net_shapes , net_params , move , weight , pool)

        # 取得testing_reward時，params不需要加noise
        testing_reward = get_reward(net_shapes , net_params , env , CONFIG['ep_max_step'] , CONFIG['continuous_a'] , None)
        
        if moving_average_reward is None:
            moving_average_reward = testing_reward
        else:
            moving_average_reward = 0.9 * moving_average_reward + 0.1 * testing_reward # moving average reward
        
        print('Generation: {} , Training_Reward : {:.2f} , Testing_Reward : {:.2f}'.format(generation , training_reward.mean() , moving_average_reward))

        if moving_average_reward >= CONFIG['eval_threshold'] :
            break

    # testing
    print('\nTesting...')
    p = params_reshape(net_shapes, net_params)
    while True:
        s = env.reset()
        for _ in range(CONFIG['ep_max_step']):
            env.render()
            a = get_action(p , s , CONFIG['continuous_a'])
            s , _ , done , _ = env.step(a)
            if done: break

