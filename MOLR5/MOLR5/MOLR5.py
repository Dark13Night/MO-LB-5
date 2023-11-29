import numpy as np

# Создание матрицы из предоставленных данных
matrix = np.array([
    [10, 9, 3, 0],
    [11, 7, 0, 15],
    [16, 6, 19, 13],
    [15, 17, 15, 10],
    [2, 1, 4, 6]
])

def bernoulli_criterion(matrix):
    # Находим стратегию, реализующую максимум математического ожидания выигрыша
    expected_values = np.mean(matrix, axis=1)
    strategy = np.argmax(expected_values)
    print("Bernoulli criterion expected values:", expected_values)
    print("Chosen strategy (1-based):", strategy + 1)
    return strategy

def wald_criterion(matrix):
    # Находим стратегию, минимизирующую риск проигрыша
    min_payoffs = np.min(matrix, axis=1)
    strategy = np.argmax(min_payoffs)
    print("Wald criterion min payoffs:", min_payoffs)
    print("Chosen strategy (1-based):", strategy + 1)
    return strategy

def optimism_criterion(matrix):
    # Находим стратегию, максимизирующую выигрыш
    max_payoffs = np.max(matrix, axis=1)
    strategy = np.argmax(max_payoffs)
    print("Optimism criterion max payoffs:", max_payoffs)
    print("Chosen strategy (1-based):", strategy + 1)
    return strategy

def gurwicz_criterion(matrix, k):
    # Находим смешанную стратегию с коэффициентом k
    min_payoffs = np.min(matrix, axis=1)
    max_payoffs = np.max(matrix, axis=1)
    mixed_strategy_payoffs = k * min_payoffs + (1 - k) * max_payoffs
    strategy = np.argmax(mixed_strategy_payoffs)
    print("Gurwicz criterion mixed strategy payoffs (k=" + str(k) + "):", mixed_strategy_payoffs)
    print("Chosen strategy (1-based):", strategy + 1)
    return strategy

def savage_criterion(matrix):
    max_payoffs = np.max(matrix, axis=0)
    risks = max_payoffs - matrix
    max_risks = np.max(risks, axis=1)  # Находим максимальные риски для каждой стратегии
    chosen_strategy = np.argwhere(max_risks == np.min(max_risks)).flatten()  # Выбираем стратегию с минимальным из максимальных связанных рисков
    print("Savage criterion max risks:", max_risks)
    print("Chosen strategy (1-based):", chosen_strategy[0] + 1)  # Выводим номер выбранной стратегии
    return chosen_strategy[0]  # Возвращаем номер выбранной стратегии

strategies = [
    bernoulli_criterion(matrix),
    wald_criterion(matrix),
    optimism_criterion(matrix),
    gurwicz_criterion(matrix, 0.5),
    savage_criterion(matrix)  # Добавим вызов функции без установки параметра
]

most_frequent_strategy = max(set(strategies), key=strategies.count)
print("Most frequent strategy (1-based):", most_frequent_strategy + 1)