import numpy as np
import matplotlib
import matplotlib.pyplot as plt

CHORD_NUM = 100                          # chord的數量
NEW_CHORD_NUM = 80                       # 欲更新chord的數量
DIM = 2                                  # 2個維度
N_GENERATION = 500                       # 共迭代500次
chord_bound = [-100 , 100]               # 限制chord的範圍
PROB_FORM_CHORD = np.array([0.2 , 0.8])  # 決定new_chord是來自於原來的chord，還是隨機產生
MUTATE_STRENGTH = 0.02 * (chord_bound[1] - chord_bound[0]) # 變種的強度
MUTATE_STRENGTH_DAMP = 0.995             # 變種強度的衰減率

#--------------------------主要函數--------------------------#
# fitness function : 找3 * (1 - x_1)**2 * np.exp(-x_1**2 - (x_2 + 1)**2) - 10 * (x_1/5 - x_1**3 - x_2**5) * np.exp(-x_1**2 - x_2**2) - 1/3 * np.exp(-(x_1 + 1)**2 - x_2**2)的最大值
def get_fitness(chord):
    finger_num = chord.shape[0]
    fitness = np.zeros(finger_num)
    for i in range(0 , finger_num):
        x_1 = chord[i , 0]
        x_2 = chord[i , 1]
        fitness[i] = 3 * (1 - x_1)**2 * np.exp(-x_1**2 - (x_2 + 1)**2) - 10 * (x_1/5 - x_1**3 - x_2**5) * np.exp(-x_1**2 - x_2**2) - 1/3 * np.exp(-(x_1 + 1)**2 - x_2**2)
    return fitness

# 畫等高線圖
def contour(ax):
    n = 300
    x = np.linspace(-3 , 3 , n)
    Y , X = np.meshgrid(x , x)
    Z = np.zeros_like(X)
    for i in range(0 , n):
        for j in range(0 , n):
            Z[i , j] = get_fitness(np.array([[x[i] , x[j]]]))
    ax[0].contourf(X , Y , Z , 50 , alpha = 0.75 , cmap = plt.cm.rainbow)
    C = ax[0].contour(X , Y , Z , 10 , alpha = 0.75 , colors = 'black')
    ax[0].clabel(C , inline = True , fontsize = 10)
    ax[0].set_ylim(-3 , 3)
    ax[0].set_xlim(-3 , 3)
#--------------------------主要函數--------------------------#

# 總共畫兩張圖
fig , ax = plt.subplots(1 , 2 , figsize = (20 , 6))
color_list = list(matplotlib.colors.cnames.values()) # 顏色列表
color_count = 0
plt.ion()
# 畫等高線圖
contour(ax)
plt.ion()

# 初始化old_chord、voice、best_voice、best_chord
old_chord = np.random.uniform(chord_bound[0] , chord_bound[1] , [CHORD_NUM , DIM])
voice = get_fitness(old_chord)
best_voice = -np.inf
best_chord = old_chord[0 , :].copy()

for generation in range(0 , N_GENERATION):
    # 生成新的chord => new_chord
    new_chord = np.zeros([NEW_CHORD_NUM , DIM])

    # 決定new_chord中的每個chord的數值
    for new_chord_index in range(0 , NEW_CHORD_NUM):
        # 每個chord中的每個維度都要看過一遍
        for dim in range(0 , DIM):
            decision_from_chord = np.random.choice(np.arange(DIM) ,
                                                   size = 1 ,
                                                   p = PROB_FORM_CHORD)[0]

            # decision_from_chord等於1的話，chord的數值來自於原來的chord
            if decision_from_chord == 1:
                index = np.random.randint(0 , CHORD_NUM , size = 1)[0]
                new_chord[new_chord_index , dim] = old_chord[index , dim]
            # decision_from_chord等於0的話，chord的數值則依據MUTATE_STRENGTH來隨機產生
            elif decision_from_chord == 0:
                new_chord[new_chord_index , dim] = old_chord[new_chord_index , dim] +\
                                                   MUTATE_STRENGTH * np.random.randn()

    # 讓new_chord在(chord_bound[0] , chord_bound[1])的範圍內
    new_chord = np.clip(new_chord , chord_bound[0] , chord_bound[1])

    # 合併原來的old_chord與新生成的new_chord => chord_merge
    # 並計算所有chord的音量 => voice_merge
    chord_merge = np.concatenate([old_chord , new_chord] , axis = 0)
    voice_merge = get_fitness(chord_merge)

    # 淘汰chord_merge中，音量較少的chord
    sort_index = voice_merge.argsort()[::-1]
    old_chord = chord_merge[sort_index , :][:CHORD_NUM, :]
    voice = voice_merge[sort_index][:CHORD_NUM]

    # 逐漸縮小變種強度
    MUTATE_STRENGTH *= MUTATE_STRENGTH_DAMP

    if voice.max() > best_voice:
        voice_copy = voice.copy()
        old_chord_copy = old_chord.copy()
        best_voice = voice[0]
        best_chord = old_chord_copy[0 , :]

    print('Generation : {} , Best_chord : {} ,  Best_Voice : {}'.format(generation , best_chord , best_voice))

    if 'sca' in globals(): sca.remove()
    sca = ax[0].scatter(old_chord[: , 0] , old_chord[: , 1] , alpha = 0.5 , s = 60 , c = 'black')
    plt.pause(0.001)

    color_count += 1
    if color_count == len(color_list): color_count = 0
    ax[1].scatter(best_chord[0] , best_chord[1] , s = 60 , color = color_list[color_count] , edgecolors = 'black')

plt.ioff()
plt.show()