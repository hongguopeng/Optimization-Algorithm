import neat
import numpy as np
import os
import gym
import multiprocessing as mp
import visualize

GAME = 'CartPole-v0'
ENV = gym.make(GAME).unwrapped
EP_STEP = 300           # maximum episode steps
GENERATION_EP = 100     # evaluate by the minimum of 11-episode rewards
TRAINING = True         # training or testing
CHECKPOINT = 9          # test on this checkpoint
config_path = os.path.join('config')

# 用multiprocessing的方式執行get_reward
# 比起用迴圈一個一個net去執行get_reward快上許多
def get_reward(net):
    # 每一個genome的net都會玩GENERATION_EP個回合的遊戲，會將每個回合的cumulative_reward紀錄在ep_reward
    # 並從這些ep_reward中取出最小的，除上EP_STEP，當作genome.fitness
    # 這裡是用木桶效應的想法來記錄genome.fitness
    # 木桶效應=>木桶盛水的多少，並不取決於桶壁上最高的那塊木塊，或全部木板的平均長度，而是取決於其中最短的那塊木板
    ep_reward = []
    for ep in range(0 , GENERATION_EP): 
        cumulative_reward = 0.
        observation = ENV.reset()
        for _ in range(0 , EP_STEP):
            action_values = net.activate(observation) # 4個input進入net，會產生2個output
            action = np.argmax(action_values)
            observation_ , reward , done , _ = ENV.step(action)
            cumulative_reward += reward
            if done : break
            observation = observation_
        ep_reward.append(cumulative_reward)
    return np.min(ep_reward) / float(EP_STEP)


def eval_genomes(genomes , config):
    # 在config中的pop_size為多少，genomes的長度就是多少
    jobs = []
    for genome_id , genome in genomes:
        net = neat.nn.RecurrentNetwork.create(genome , config)
        jobs.append(POOL.apply_async(get_reward , [net]))

    for job , (genome_id , genome) in zip(jobs , genomes):
        genome.fitness = job.get()


def evaluation():
    pop_test = neat.Checkpointer.restore_checkpoint('neat-checkpoint-{}'.format(CHECKPOINT))

    # find the winner in restored population
    winner = pop_test.run(eval_genomes , 1)

    # show winner net
    node_names = {-1 : 'In0' , -2 : 'In1' , -3 : 'In3' , -4 : 'In4' , 0 : 'act1' , 1 : 'act2'}
    visualize.draw_net(pop_test.config , winner , True ,  node_names = node_names)

    net = neat.nn.RecurrentNetwork.create(winner , pop_test.config)
    while True:
        s = ENV.reset()
        while True:
            ENV.render()
            a = np.argmax(net.activate(s))
            s , r , done, _ = ENV.step(a)
            if done : break


if __name__ == "__main__":
    if TRAINING:
        config = neat.Config(neat.DefaultGenome ,
                             neat.DefaultReproduction ,
                             neat.DefaultSpeciesSet ,
                             neat.DefaultStagnation , 
                             config_path)
        pop_train = neat.Population(config)
    
        stats = neat.StatisticsReporter()
        pop_train.add_reporter(stats)
        pop_train.add_reporter(neat.StdOutReporter(True))
        pop_train.add_reporter(neat.Checkpointer(5))

        # 訓練10個generation
        POOL = mp.Pool(processes = 4)
        pop_train.run(eval_genomes , 10)

    else:
        evaluation()