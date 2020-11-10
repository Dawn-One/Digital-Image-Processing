import numpy as np
import matplotlib.pyplot as plt


def his_equal(prk):

    """make the histogram equalization"""

    mapping = np.zeros(prk.shape, [('x', int), ('y', int), ('z', float)])   # 创建空映射
    mapping['z'] = prk
    mapping['x'] = np.arange(len(prk))

    # 画图
    plt.subplot(221)
    plt.plot(mapping['x'], mapping['z'], 'bo')
    for i in range(len(mapping)):
        plt.plot([mapping['x'][i], mapping['x'][i]], [0, mapping['z'][i]], 'b--')
    plt.ylim(0, max(mapping['z']+0.05))

    # 计算s
    for i in range(len(prk)):
        mapping['y'][i] = round((len(prk) - 1) * np.add.reduce(prk[0:i + 1]))

    pss = np.zeros(len(np.unique(mapping['y'])), [('x', int), ('y', float)])  # 输出s的下标和p(s)
    pss['x'] = np.unique(mapping['y'])

    # 计算不同s的p(s),相同的s值的项的p(s)加到一起
    for i in range(len(pss)):
        for j in range(len(prk)):
            if pss['x'][i] == mapping['y'][j]:
                pss['y'][i] += mapping['z'][j]

    return mapping, pss


def his_define(map, pss):

    """ Normalize histogram """

    his_target_p = np.array([0, 0, 0, 0.15, 0.20, 0.30, 0.20, 0.15])        # 目标直方图的值
    his_target = np.zeros(len(his_target_p), [('x', int), ('y', float)])
    his_target['x'] = np.arange(len(his_target_p))
    his_target['y'] = his_target_p

    # 画图
    plt.subplot(222)
    plt.plot(his_target['x'], his_target['y'], 'bo')
    for i in range(len(his_target)):
        plt.plot([his_target['x'][i], his_target['x'][i]], [0, his_target['y'][i]], 'b--')
    plt.ylim(0, max(his_target['y']+0.05))

    # 计算G值
    g = np.zeros(len(map))
    for i in range(len(his_target)):
        g[i] = round((len(map) - 1) * np.add.reduce(his_target['y'][0:i+1]))
    print(g)
    plt.subplot(223)
    plt.step(np.arange(g.shape[0]), g)

    # 寻找s和z的映射关系
    sz_map = np.zeros(len(pss), [('x', int), ('y', int), ('z', float)])
    sz_map['x'] = pss['x']
    for i in range(len(sz_map)):
        sz_map['y'][i] = abs(sz_map['x'][i] - g).argmin()
        sz_map['z'][i] = pss['y'][i]

    plt.subplot(224)
    plt.plot(sz_map['y'], sz_map['z'], 'bo')
    for i in range(len(sz_map)):
        plt.plot([sz_map['y'][i], sz_map['y'][i]], [0, sz_map['z'][i]], 'b--')
    plt.ylim(0, max(sz_map['z']+0.05))
    plt.xlim(0, 7)

    return sz_map


prk = np.array([0.19, 0.25, 0.21, 0.16, 0.08, 0.06, 0.03, 0.02])    # 源图像直方图数据
s, pss = his_equal(prk)         # 直方图均衡化
print(pss)
normalize_map = his_define(s, pss)              # 直方图规定化
print('s->z: ', normalize_map)
plt.show()
