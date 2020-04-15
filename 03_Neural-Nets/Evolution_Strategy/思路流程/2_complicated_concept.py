'''
每一次iteration都會計算reward的平均值，會發現迭代前期reward的平均值會一直上升
但是reward的平均值最後會卡住，無法再提升
猜測是與noise的生成有關，因為noise的sigma在每次的迭代中都不會改變
導致最後無法精準的sample到可以讓reward的平均值持續下降的noise
因此在 "3_complicated_concept_改良版" 這支程式中
每次的迭代過程中，只要符合條件就讓sigma越來越小
期望能sample到讓reward的平均值持續下降的noise
'''
import numpy as np

in_ = np.array([1 , 1 , 1 , 1]) # 輸入neural_network的值
target = np.array([5 , 2 , 3])  # 想要預測的答案
n_scout = 50      # 斥候人數
sigma = 0.1       # noise的標準差
alpha = 0.001     # 學習率

#--------------------------主要函數--------------------------#
def f(pred):
    return -np.sum((pred - target) ** 2)


def build_net():
    def linear(neuron_in , neuron_out):
        weight = np.random.randn(neuron_in * neuron_out).astype(np.float32) # 得到初始weight
        bias = np.random.randn(neuron_out).astype(np.float32)               # 得到初始bias
        shape = [neuron_in , neuron_out]
        param = np.concatenate([weight , bias])
        return shape , param

    shape_0 , param_0 = linear(4 , 5)  # input param
    shape_1 , param_1 = linear(5 , 4)  # hidden param
    shape_2 , param_2 = linear(4 , 3)  # output param
    shapes_set = [shape_0 , shape_1 , shape_2]
    params_set = np.concatenate([param_0 , param_1 , param_2])
    return shapes_set , params_set , np.zeros_like(params_set).astype(np.float32)


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


def neural_network(params , in_):
    in_ = in_.reshape([1 , len(in_)])          # input layer
    h_1 = np.dot(in_ , params[0]) + params[1]  # hidden layer_1
    h_1 = np.maximum(0 , h_1)                  # 將h_1丟進activaction function=>ReLU
    h_2 = np.dot(h_1 , params[2]) + params[3]  # hidden layer_2
    h_2 = np.maximum(0 , h_2)                  # 將h_2丟進activaction function=>ReLU
    pred = np.dot(h_2 , params[4]) + params[5] # output layer
    pred = pred.reshape([-1 , ])
    return pred
#--------------------------主要函數--------------------------#


# 得到初始的net_params，而net_params包含weight與bias
# 但net_params只是1維陣列，最後再配合net_shapes，將net_params包含weight與bias重新整理成原來matrix的形式
net_shapes , net_params , move = build_net()

for iteration in range(0 , 10000):

    reward = np.zeros(n_scout)
    noise = np.random.randn(n_scout , net_params.shape[0]) * sigma  # 產生noise
    net_params_perturb = net_params + noise # 對net_params加一些擾動，得到net_params_perturb

    for scout in range(0 , n_scout):
        # 將net_params包含weight與bias重新整理成原來matrix的形式
        net_params_perturb_reshape = params_reshape(net_shapes , net_params_perturb[scout , :])

        # 將in_丟進NN，得到預測值pred
        pred = neural_network(net_params_perturb_reshape , in_)

        # 利用pred得到reward，也就是計算pred與target有多少差距
        reward[scout] = f(pred)

    if iteration % 10 == 0:
        print('iter : {} , mean_reward : {:.3f} , pred : {}'.format(iteration , reward.mean() , pred))

    # 對reward做normalization
    normalized_reward = (reward - np.mean(reward)) / np.std(reward)
    '''
                                                              50
    以normalized_reward當作權重，對所有noise[i , :]做加權總合 => ∑(noise[i , :] * normalized_reward[i])
                                                             i=0
    並不是放棄不是很大reward所對應的noise，而是將所有noise列入考慮                                                         
    '''
    noise_weghted_sum = np.dot(noise.T , normalized_reward)

    # move => 9成的比例遵循原來move的方向，另外1成遵循noise_weghted_sum的方向
    momentum = 0.9
    move = momentum * move + (1. - momentum) * noise_weghted_sum / (n_scout * sigma)

    # 更新net_params
    net_params = net_params + alpha * move
