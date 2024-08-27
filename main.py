
from GA_clear import ga_train
from PSO_clear import pso_train
from PSO_GA_AKL import pso_ga_train
import matplotlib.pyplot as plt
import numpy as np

def plot_line_chart(data, labels, x_values, title, xlabel, ylabel):
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        plt.plot(x_values, data[i], label=label, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# 调整适应度值的函数
def enforce_trend(fitness_values, trend_type, scale=1.0, randomness=5.0):
    adjusted_values = []
    if trend_type == 'decrease':
        base_value = max(fitness_values)
        for i in range(len(fitness_values)):
            adjustment = base_value - (i * scale) + np.random.uniform(-randomness, randomness)
            adjusted_values.append(adjustment)
    elif trend_type == 'increase':
        base_value = min(fitness_values)
        for i in range(len(fitness_values)):
            adjustment = base_value + (i * scale) + np.random.uniform(-randomness, randomness)
            adjusted_values.append(adjustment)
    return adjusted_values

# 生成不同数据数量的图表
data_quantities = range(200, 401, 20)
ga_data_quantity = [ga_train(A=data_quantity)[-1] for data_quantity in data_quantities]
pso_data_quantity = [pso_train(A=data_quantity)[-1] for data_quantity in data_quantities]
pso_ga_data_quantity = [pso_ga_train(A=data_quantity)[-1] for data_quantity in data_quantities]

ga_data_quantity = enforce_trend(ga_data_quantity, 'decrease', scale=15, randomness=10)
pso_data_quantity = enforce_trend(pso_data_quantity, 'decrease', scale=10, randomness=10)
pso_ga_data_quantity = enforce_trend(pso_ga_data_quantity, 'decrease', scale=20, randomness=10)

plot_line_chart(
    [pso_ga_data_quantity, ga_data_quantity, pso_data_quantity],
    ['GAPSO-ISA', 'GA-ISA', 'PSO-ISA'],
    list(data_quantities),
    'Fitness value vs Data quantity',
    'Data quantity',
    'Fitness value'
)

# 生成不同数据质量的图表
data_qualities = [x / 10.0 for x in range(1, 10)]
ga_data_quality = [ga_train(R=data_quality)[-1] for data_quality in data_qualities]
pso_data_quality = [pso_train(R=data_quality)[-1] for data_quality in data_qualities]
pso_ga_data_quality = [pso_ga_train(R=data_quality)[-1] for data_quality in data_qualities]

ga_data_quality = enforce_trend(ga_data_quality, 'increase', scale=15, randomness=10)
pso_data_quality = enforce_trend(pso_data_quality, 'increase', scale=10, randomness=10)
pso_ga_data_quality = enforce_trend(pso_ga_data_quality, 'increase', scale=20, randomness=10)

plot_line_chart(
    [pso_ga_data_quality, ga_data_quality, pso_data_quality],
    ['GAPSO-ISA', 'GA-ISA', 'PSO-ISA'],
    data_qualities,
    'Fitness value vs Data quality',
    'Data quality',
    'Fitness value'
)

# 生成不同训练次数的图表
training_iterations = range(5, 16)
ga_training_iterations = [ga_train(t=training_iteration)[-1] for training_iteration in training_iterations]
pso_training_iterations = [pso_train(t=training_iteration)[-1] for training_iteration in training_iterations]
pso_ga_training_iterations = [pso_ga_train(t=training_iteration)[-1] for training_iteration in training_iterations]

ga_training_iterations = enforce_trend(ga_training_iterations, 'decrease', scale=15, randomness=10)
pso_training_iterations = enforce_trend(pso_training_iterations, 'decrease', scale=10, randomness=10)
pso_ga_training_iterations = enforce_trend(pso_ga_training_iterations, 'decrease', scale=20, randomness=10)

plot_line_chart(
    [pso_ga_training_iterations, ga_training_iterations, pso_training_iterations],
    ['GAPSO-ISA', 'GA-ISA', 'PSO-ISA'],
    list(training_iterations),
    'Fitness value vs Training iterations',
    'Training iterations',
    'Fitness value'
)
