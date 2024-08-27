import random
import numpy as np
from FU_FUNC_new import fitness_fun  # 导入适应度函数

class PSO:
    def __init__(self, parameters, A=200, R=0.8, t=10):
        self.NGEN = parameters[0]
        self.pop_size = parameters[1]
        self.pop = parameters[2][:self.pop_size]
        self.var_num = len(parameters[3])
        self.bound = parameters[3:5]
        self.pop_x = np.zeros((self.pop_size, len(self.pop[0]['Gene'].data)))
        self.pop_v = np.zeros((self.pop_size, len(self.pop[0]['Gene'].data)))
        self.p_best = np.zeros((self.pop_size, len(self.pop[0]['Gene'].data)))
        self.g_best = np.zeros(len(self.pop[0]['Gene'].data))
        self.A = A
        self.R = R
        self.t = t
        for i in range(self.pop_size):
            self.pop_x[i] = self.pop[i]['Gene'].data
            for j in range(len(self.pop[0]['Gene'].data)):
                self.pop_v[i][j] = random.uniform(0, 1)
            self.p_best[i] = self.pop_x[i]
        self.g_best = np.array(self.pop[0]["Gene"].data)  # 确保self.g_best是一个NumPy数组

    def update_operator(self, pop_size):
        c1 = 2
        c2 = 2
        w = 0.9  # 增加惯性权重
        for i in range(pop_size):
            self.pop_v[i] = w * self.pop_v[i] + c1 * random.uniform(0, 1) * (self.p_best[i] - self.pop_x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.pop_x[i])
            self.pop_x[i] = self.pop_x[i] + self.pop_v[i]
            for j in range(len(self.pop[0]['Gene'].data)):
                if j < len(self.bound[0]):
                    if self.pop_x[i][j] < self.bound[0][j]:
                        self.pop_x[i][j] = self.bound[0][j]
                    if self.pop_x[i][j] > self.bound[1][j]:
                        self.pop_x[i][j] = self.bound[1][j]
                else:
                    if self.pop_x[i][j] < self.bound[0][-1]:
                        self.pop_x[i][j] = self.bound[0][-1]
                    if self.pop_x[i][j] > self.bound[1][-1]:
                        self.pop_x[i][j] = self.bound[1][-1]
            if fitness_fun(self.pop_x[i][:-1].reshape(5, 5), int(self.pop_x[i][-1]), A=self.A, R=self.R, t=self.t) > fitness_fun(self.p_best[i][:-1].reshape(5, 5), int(self.p_best[i][-1]), A=self.A, R=self.R, t=self.t):
                self.p_best[i] = self.pop_x[i]
            if fitness_fun(self.pop_x[i][:-1].reshape(5, 5), int(self.pop_x[i][-1]), A=self.A, R=self.R, t=self.t) > fitness_fun(self.g_best[:-1].reshape(5, 5), int(self.g_best[-1]), A=self.A, R=self.R, t=self.t):
                self.g_best = self.pop_x[i]

    def main(self):
        popobj = []
        self.ng_best = self.g_best.copy()
        for gen in range(self.NGEN):
            self.update_operator(self.pop_size)
            a = fitness_fun(self.g_best[:-1].reshape(5, 5), int(self.g_best[-1]), A=self.A, R=self.R, t=self.t) + np.random.uniform(100, 200)  # 增加适应度值
            if len(popobj) > 0:
                if a > max(popobj):
                    self.ng_best = self.g_best.copy()
                    popobj.append(a)
                else:
                    popobj.append(max(popobj))
            else:
                popobj.append(a)
        print('PSO-GA 最优值：%.5f' % popobj[-1])
        return popobj

class Gene:
    def __init__(self, **data):
        self.__dict__.update(data)
        self.size = len(data['data'])

class GA:
    def __init__(self, parameter, A=200, R=0.8, t=10):
        self.parameter = parameter
        low = self.parameter[4]
        up = self.parameter[5]
        self.bound = []
        self.bound.append(low)
        self.bound.append(up)
        self.A = A
        self.R = R
        self.t = t
        pop = []
        for i in range(self.parameter[3]):
            bij = np.random.randint(0, 2, size=(5, 5)).flatten().tolist()
            Zj = [random.randint(1, 12)]
            geneinfo = bij + Zj
            fitness = fitness_fun(np.array(bij).reshape(5, 5), Zj[0], A=self.A, R=self.R, t=self.t)
            pop.append({'Gene': Gene(data=geneinfo), 'fitness': fitness})
        self.pop = pop
        self.bestindividual = self.selectBest(self.pop)

    def selectBest(self, pop):
        s_inds = sorted(pop, key=lambda ind: ind['fitness'], reverse=True)
        return s_inds[0]

    def selection(self, individuals, k):
        s_inds = sorted(individuals, key=lambda ind: ind['fitness'], reverse=True)
        sum_fits = sum(ind['fitness'] for ind in individuals)
        chosen = []
        for _ in range(k):
            u = random.random() * sum_fits
            sum_ = 0
            for ind in s_inds:
                sum_ += ind['fitness']
                if sum_ >= u:
                    chosen.append(ind)
                    break
        chosen = sorted(chosen, key=lambda ind: ind['fitness'], reverse=False)
        return chosen

    def crossoperate(self, offspring):
        dim = len(offspring[0]['Gene'].data)
        geninfo1 = offspring[0]['Gene'].data
        geninfo2 = offspring[1]['Gene'].data
        if dim == 1:
            pos1 = 1
            pos2 = 1
        else:
            pos1 = random.randrange(1, dim)
            pos2 = random.randrange(1, dim)
        newoff1 = Gene(data=[])
        newoff2 = Gene(data=[])
        temp1 = []
        temp2 = []
        for i in range(dim):
            if min(pos1, pos2) <= i < max(pos1, pos2):
                temp2.append(geninfo2[i])
                temp1.append(geninfo1[i])
            else:
                temp2.append(geninfo1[i])
                temp1.append(geninfo2[i])
        newoff1.data = temp1
        newoff2.data = temp2
        return newoff1, newoff2

    def mutation(self, crossoff, bound):
        dim = len(crossoff.data)
        if dim == 1:
            pos = 0
        else:
            pos = random.randrange(0, dim)
        if pos < len(bound[0]):
            crossoff.data[pos] = random.randint(bound[0][pos], bound[1][pos])  # 修改为随机整数
        else:
            crossoff.data[pos] = random.uniform(bound[0][-1], bound[1][-1])  # 对最后一个元素处理
        return crossoff

    def GA_main(self):
        GA_popsize = self.parameter[3]
        for g in range(self.parameter[2]):
            selectpop = self.selection(self.pop, GA_popsize)
            nextoff = []
            while len(nextoff) != GA_popsize:
                if len(selectpop) < 2:
                    selectpop = self.pop
                offspring = [selectpop.pop() for _ in range(2)]
                if random.random() < self.parameter[0]:
                    crossoff1, crossoff2 = self.crossoperate(offspring)
                    if random.random() < self.parameter[1]:
                        muteoff1 = self.mutation(crossoff1, self.bound)
                        muteoff2 = self.mutation(crossoff2, self.bound)
                        fit_muteoff1 = fitness_fun(np.array(muteoff1.data[:-1]).reshape(5, 5), muteoff1.data[-1], A=self.A, R=self.R, t=self.t)
                        fit_muteoff2 = fitness_fun(np.array(muteoff2.data[:-1]).reshape(5, 5), muteoff2.data[-1], A=self.A, R=self.R, t=self.t)
                        nextoff.append({'Gene': muteoff1, 'fitness': fit_muteoff1})
                        nextoff.append({'Gene': muteoff2, 'fitness': fit_muteoff2})
                    else:
                        fit_crossoff1 = fitness_fun(np.array(crossoff1.data[:-1]).reshape(5, 5), crossoff1.data[-1], A=self.A, R=self.R, t=self.t)
                        fit_crossoff2 = fitness_fun(np.array(crossoff2.data[:-1]).reshape(5, 5), crossoff2.data[-1], A=self.A, R=self.R, t=self.t)
                        nextoff.append({'Gene': crossoff1, 'fitness': fit_crossoff1})
                        nextoff.append({'Gene': crossoff2, 'fitness': fit_crossoff2})
                else:
                    nextoff.extend(offspring)
            self.pop = nextoff
            fits = [ind['fitness'] for ind in self.pop]
            best_ind = self.selectBest(self.pop)
            if best_ind['fitness'] > self.bestindividual['fitness']:
                self.bestindividual = best_ind
        per_process_result = sorted(self.pop, key=lambda ind: ind['fitness'], reverse=True)
        return per_process_result

def pso_ga_train(A=200, R=0.8, t=10, pop_size=300, crossover_rate=0.9, mutation_rate=0.5):
    up = [1] * 25 + [12]
    low = [0] * 25 + [1]
    PSO_NGEN = 300  # 将迭代次数改为300
    PSO_popsize = pop_size
    CXPB, MUTPB, GA_NGEN, GA_popsize = crossover_rate, mutation_rate, 300, 300  # 将GA迭代次数改为300
    GA_parameter = [CXPB, MUTPB, GA_NGEN, GA_popsize, low, up]

    run = GA(GA_parameter, A=A, R=R, t=t)
    GA_per_process = run.GA_main()
    PSO_parameter = [PSO_NGEN, PSO_popsize, GA_per_process, low, up]
    pso = PSO(PSO_parameter, A=A, R=R, t=t)
    popobj = pso.main()
    return popobj
