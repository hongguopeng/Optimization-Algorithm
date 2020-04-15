import numpy as np
import gym
import multiprocessing as mp
import time

N_KID = 20                   # half of the training population
N_GENERATION = 5000          # training step
LR = 0.05                    # learning rate
SIGMA = 0.05                 # mutation strength or step size
N_CORE = mp.cpu_count() - 1
MOMENTUM = 0.9
CONFIG = [dict(game = "CartPole-v0",
               n_feature = 4 , 
               n_action = 2 ,
               continuous_a = [False] ,
               ep_max_step = 700 ,
               eval_threshold = 650) ,
          
          dict(game = "MountainCar-v0" ,
               n_feature = 2 ,
               n_action = 3 ,
               continuous_a = [False] ,
               ep_max_step = 200 , 
               eval_threshold = -120) ,
          
          dict(game = "Pendulum-v0" ,
               n_feature = 3 ,
               n_action = 1 ,
               continuous_a = [True , 2.] , 
               ep_max_step = 200 ,
               eval_threshold = -180)    ][0]    # choose your game


def params_reshape(shapes , params):   
    w_b_list = [[] for _ in range(0 , len(shapes) * 2)]
    count = 0
    for i in range(0 , len(shapes) * 2 , 2):
        w_b_list[i] = shapes[count]
        w_b_list[i + 1] = shapes[count][1]
        count += 1

    p = []
    for i in range(0 , len(w_b_list)):
        if i % 2 == 0.:
            idx = w_b_list[i][0] * w_b_list[i][1]
            temp = params[0 : idx].reshape(w_b_list[i][0] , w_b_list[i][1])
            params = np.delete(params , range(0 , idx)) 
        elif i % 2 == 1.:
            temp = params[0 : w_b_list[i]].reshape(1 , w_b_list[i])
            params = np.delete(params , range(0 , w_b_list[i])) 
        p.append(temp)
         
    return p


def get_action(params , x , continuous_a):
    x = x[np.newaxis, :]
    x = np.tanh(x.dot(params[0]) + params[1])
    x = np.tanh(x.dot(params[2]) + params[3])
    x = x.dot(params[4]) + params[5]
    if not continuous_a[0]: 
        return np.argmax(x , axis = 1)[0]      
    else: 
        return continuous_a[1] * np.tanh(x)[0] 


def get_reward_mp(sigma , shapes , params , env , ep_max_step , continuous_a):
    perturb = np.random.randn(params.shape[0])
    params = params + sigma * perturb 
    p = params_reshape(shapes , params)
                    
    # run episode
    s = env.reset()
    ep_r = 0.
    for step in range(0 , ep_max_step):
        a = get_action(p , s , continuous_a)
        s , r , done , _ = env.step(a)
        # mountain car's reward can be tricky
        if env.spec._env_name == 'MountainCar' and s[0] > -0.1: 
            r = 0.
        ep_r += r
        if done: break
    return [ep_r , perturb]


class ESNN(object):
    def __init__(self , n_kid , lr , sigma , n_core , config , momentum):
        self.n_kid = n_kid 
        self.lr = lr
        self.sigma = sigma
        self.n_core = n_core
        self.config = config
        self.momentum = momentum
        
        rank = np.arange(1 , self.n_kid + 1)
        weight = np.maximum(0, np.log(self.n_kid / 2 + 1) - np.log(rank))
        self.weight = weight / weight.sum() - 1 / self.n_kid
        
        self.env = gym.make(self.config['game']).unwrapped
        self.pool = mp.Pool(processes = self.n_core) 
    
    
    @staticmethod    
    def linear(n_in , n_out):  
        np.random.seed(100)
        w = np.random.randn(n_in * n_out).astype(np.float32) * 0.1
        b = np.random.randn(n_out).astype(np.float32) * 0.1
        return (n_in , n_out) , np.concatenate((w , b))
        
    
    def build_net(self):
        s0 , p0 = self.linear(self.config['n_feature'] , 30)
        s1 , p1 = self.linear(30 , 20)
        s2 , p2 = self.linear(20 , CONFIG['n_action'])
        self.shapes = [s0 , s1 , s2]
        self.net_params = np.concatenate((p0 , p1 , p2))  
        self.v = np.zeros_like(self.net_params).astype(np.float32) 
     
        
    def train(self):
        jobs = []
        for k_id in range(0 , self.n_kid):
            # 若要用多核心執行get_reward_mp這個函數，切記get_reward_mp這個函數不能是ESNN中的一個類別!!
            # 以ESNN去繼承其他class，再用多核心執行class中的get_reward_mp這個函數也不行!!
            jobs.append(self.pool.apply_async(get_reward_mp , (self.sigma , 
                                                               self.shapes , 
                                                               self.net_params , 
                                                               self.env , 
                                                               self.config['ep_max_step'], 
                                                               self.config['continuous_a'])))
        rewards , perturb = [] , []
        for j in jobs:
            temp = j.get()
            rewards.append(temp[0])
            perturb.append(temp[1])
        self.rewards = np.array(rewards) 
        self.perturb = np.array(perturb) 
            
        self.kids_rank = np.argsort(self.rewards)[::-1]              
    
        self.cumulative_update = np.zeros_like(self.net_params)       
        for ui , k_id in enumerate(self.kids_rank):
            self.cumulative_update = self.cumulative_update + self.weight[ui] * self.perturb[k_id , :]  
                
        self.v = self.momentum * self.v + (1. - self.momentum) * self.cumulative_update / (self.n_kid * self.sigma) 
        self.gradients = self.lr * self.v  
        
        self.net_params = self.net_params + self.gradients
        self.kid_rewards = self.rewards
        
        
if __name__ == "__main__":    
    esnn = ESNN(n_kid = N_KID , 
                lr = LR , 
                sigma = SIGMA , 
                n_core = N_CORE , 
                config = CONFIG ,
                momentum = MOMENTUM)
    
    esnn.build_net()
    mar = None 
    for g in range(0 , N_GENERATION):
        t0 = time.time()
        esnn.train()
        [net_r , _] = get_reward_mp(esnn.sigma , esnn.shapes , esnn.net_params , esnn.env , esnn.config['ep_max_step'] , esnn.config['continuous_a'])
        
        if mar is None:
            mar = net_r
        else:
            mar = 0.9 * mar + 0.1 * net_r # moving average reward    
        
        print('Gen: ', g,
              '| Net_R: %.1f' % mar,
              '| Kid_avg_R: %.1f' % esnn.kid_rewards.mean(),
              '| Gen_T: %.2f' % (time.time() - t0))
        
        if mar >= esnn.config['eval_threshold'] : 
            break    
        
        
        