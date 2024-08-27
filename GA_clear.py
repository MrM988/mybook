import numpy as np
from FU_FUNC_new import fitness_fun  # 导入适应度函数

DNA_SIZE = 25
CROSSOVER_RATE = 0.6  # 增加交叉率
MUTATION_RATE = 0.001  # 增加变异概率
N_GENERATIONS = 300
Z_BOUND = [1, 12]

def get_fitness(pop, A=200, R=0.8, t=10):
    bij, Zj = translateDNA(pop)
    pred = np.array([fitness_fun(bij[i].reshape(5, 5), Zj[i], A=A, R=R, t=t) + np.random.uniform(-30, 30) for i in range(len(bij))])  # 减小噪声幅度
    return pred

def translateDNA(pop):
    bij_pop = pop[:, :-1].reshape(-1, 5, 5)
    Zj_pop = pop[:, -1]
    Zj = Zj_pop * (Z_BOUND[1] - Z_BOUND[0]) + Z_BOUND[0]
    return bij_pop, Zj

def crossover_and_mutation(pop, CROSSOVER_RATE=0.6):  # 增加交叉率
    new_pop = []
    for father in pop:
        child = father.copy()
        if np.random.rand() < CROSSOVER_RATE:
            mother = pop[np.random.randint(len(pop))]
            cross_points = np.random.randint(low=0, high=DNA_SIZE)
            child[cross_points:] = mother[cross_points:]
        mutation(child)
        new_pop.append(child)
    return np.array(new_pop)

def mutation(child, MUTATION_RATE=0.0005):  # 增加变异概率
    if np.random.rand() < MUTATION_RATE:
        mutate_point = np.random.randint(0, DNA_SIZE)
        child[mutate_point] = child[mutate_point] ^ 1

def select(pop, fitness):
    fitness = np.array(fitness)
    fitness = np.nan_to_num(fitness, nan=0.1, posinf=0.1, neginf=0.1)
    fitness[fitness < 0] = 0.1
    idx = np.random.choice(np.arange(len(pop)), size=len(pop), replace=True, p=(fitness) / (fitness.sum()))
    return pop[idx]

def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print("GA max_fitness:", fitness[max_fitness_index])
    bij, Zj = translateDNA(pop)
    print("GA 最优的基因型：", pop[max_fitness_index])
    print("(bij, Zj):", (bij[max_fitness_index].reshape(5, 5), Zj[max_fitness_index]))

def ga_train(pop_size=30, crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE, A=200, R=0.8, t=10):
    fitness_list_res = []
    pop = np.random.randint(2, size=(pop_size, DNA_SIZE + 1))
    for _ in range(N_GENERATIONS):
        bij, Zj = translateDNA(pop)
        pop = np.array(crossover_and_mutation(pop, crossover_rate))
        fitness = get_fitness(pop, A=A, R=R, t=t)
        pop = select(pop, fitness)
        if len(fitness_list_res) > 0:
            a = fitness.max()
            fitness_list_res.append(max(a, max(fitness_list_res)))
        else:
            fitness_list_res.append(fitness.max())
    print_info(pop)
    return fitness_list_res


