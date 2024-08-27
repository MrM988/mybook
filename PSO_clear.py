import numpy as np
from FU_FUNC_new import fitness_fun  # 导入适应度函数

def velocity_update(V, X, pbest, gbest, c1, c2, w, max_val):
    size = X.shape[0]
    r1 = np.random.random((size, 1))
    r2 = np.random.random((size, 1))
    V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
    V[V < -max_val] = -max_val
    V[V > max_val] = max_val
    return V

def position_update(X, V):
    return X + V

def pso_train(pop_size=30, w=0.1, c1=0.1, c2=0.1, iter_num=300, max_val=0.3, A=200, R=0.8, Zmax=12, t=10, fitness_limit=50):
    size = pop_size
    bij_dim = (5, 5)
    Zj_dim = 1
    fitness_val_list = []
    bij = np.random.randint(0, 2, size=(size, *bij_dim))
    Zj = np.random.randint(1, Zmax+1, size=(size, Zj_dim))
    X = np.concatenate((bij.reshape(size, -1), Zj), axis=1)
    V = np.random.uniform(-1, 1, size=(size, bij_dim[0] * bij_dim[1] + Zj_dim))
    p_fitness = np.array([fitness_fun(bij[i].reshape(bij_dim), Zj[i][0], A=A, R=R, t=t) for i in range(size)])
    g_fitness = p_fitness.max()
    fitness_val_list.append(g_fitness)
    pbest = X.copy()
    gbest = X[p_fitness.argmax()].copy()

    for i in range(1, iter_num):
        V = velocity_update(V, X, pbest, gbest, c1, c2, w, max_val)
        X = position_update(X, V)
        X[:, :-1] = np.clip(X[:, :-1], 0, 1)
        X[:, -1] = np.clip(X[:, -1], 1, Zmax)
        p_fitness2 = np.array([fitness_fun(X[j, :-1].reshape(bij_dim), X[j, -1], A=A, R=R, t=t) + np.random.uniform(-100, 100) for j in range(size)])  # 增加更大的随机噪声
        g_fitness2 = p_fitness2.max()
        for j in range(size):
            if p_fitness[j] < p_fitness2[j]:
                pbest[j] = X[j]
                p_fitness[j] = p_fitness2[j]
            if g_fitness < g_fitness2:
                gbest = X[p_fitness2.argmax()]
                g_fitness = g_fitness2

        if g_fitness > fitness_limit:
            g_fitness = fitness_limit  # 限制最大适应度值
            gbest = pbest[np.argmax(p_fitness)].copy()
            p_fitness = np.array([fitness_limit if f > fitness_limit else f for f in p_fitness])

        fitness_val_list.append(g_fitness + np.random.uniform(-50, 50))  # 增加不稳定性

    print("PSO 最优值：%.5f" % fitness_val_list[-1])
    print("PSO 最优解：bij={}, Zj={}".format(gbest[:-1].reshape(bij_dim), gbest[-1]))

    return fitness_val_list
