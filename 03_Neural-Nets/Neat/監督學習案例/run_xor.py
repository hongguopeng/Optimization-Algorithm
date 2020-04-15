import os
import neat
import visualize
import numpy as np

xor_inputs = [(0.0 , 0.0) , (0.0 , 1.0) , (1.0 , 0.0) , (1.0 , 1.0)]
xor_outputs = [      0.0  ,        1.0  ,        1.0  ,        0.0]


def eval_genomes(genomes , config):
    # 在config中的pop_size為多少，genomes的長度就是多少 
    for genome_id , genome in genomes:
        # 創建genome對應的net
        net = neat.nn.FeedForwardNetwork.create(genome , config)
        fitness = 0
        for i in range(0 , 4):
            output = net.activate(xor_inputs[i])
            fitness += (output[0] - xor_outputs[i]) ** 2
        genome.fitness = -fitness


# Load configuration
config_path = os.path.join('config')
# 根據配置文件創建種群
config = neat.Config(neat.DefaultGenome ,
                     neat.DefaultReproduction,
                     neat.DefaultSpeciesSet ,
                     neat.DefaultStagnation,
                     config_path)

pop_train = neat.Population(config)


stats = neat.StatisticsReporter()
pop_train.add_reporter(stats)
pop_train.add_reporter(neat.StdOutReporter(True))
# 每5次迭代生成一個checkpoint
pop_train.add_reporter(neat.Checkpointer(5))

# 訓練300個generation
winner = pop_train.run(eval_genomes , 300)

# 展示最好的genome
print('\nBest genome : \n{}'.format(winner))

# 展示在最好的genome下的訓練結果
winner_net = neat.nn.FeedForwardNetwork.create(winner , config)
print('\n')
for i in range(0 , 4):
    output = winner_net.activate(xor_inputs[i])
    print('input : {} , target : {} , predict : {}'.format(xor_inputs[i] , xor_outputs[i] , output))

# 用於展示net時輸入節點和輸出節點的編號處理
# input node從-1、-2、-3...編號，output node從0、1、2...編號
# 2個input、1個output，在config中的"network parameters部分"中輸入 num_inputs為2、num_outputs為1
node_names = {-1 : 'A' , -2 : 'B' , 0 : 'A XOR B'}
# 繪製net
visualize.draw_net(config , winner , True , node_names = node_names)
# 繪制最優與平均適應度，ylog表示y軸使用symmetric log的scale
visualize.plot_stats(stats , ylog = False , view = True)
# 可視化種群變化
visualize.plot_species(stats , view = True)


# 讀取訓練好的參數
files = os.listdir('.')
files = [file for file in files if 'neat-checkpoint' in file]
# 找出最後一個生成的neat-checkpoint的檔案
max_num = -np.inf
for f in files:
    if int(f.split('-')[-1]) > max_num:
        max_num = int(f.split('-')[-1])



#------------------------testing階段------------------------#
pop_test = neat.Checkpointer.restore_checkpoint('neat-checkpoint-{}'.format(max_num))
winner_test = pop_test.run(eval_genomes , 10)
winner_net_test = neat.nn.FeedForwardNetwork.create(winner_test , config)
print('\n')
for i in range(0 , 4):
    output = winner_net_test.activate(xor_inputs[i])
    print('input : {} , target : {} , predict : {}'.format(xor_inputs[i] , xor_outputs[i] , output))