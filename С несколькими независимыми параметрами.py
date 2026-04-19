from random import randint, uniform
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Генерация данных
n_samples = 50
x1_values = [randint(28, 95) for _ in range(n_samples)]  # Площадь квартиры (м²)
x2_values = [randint(1, 15) for _ in range(n_samples)]   # Расстояние до центра (км)
y_values = [0.8 * x1 - 3 * x2 + uniform(-5, 5) for x1, x2 in zip(x1_values, x2_values)]  # Стоимость

# Коэффициенты регрессии (начальные случайные значения)
lr = 0.001
b, k1, k2 = uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)

# Функция предсказания стоимости
def f(x1, x2):
    return k1 * x1 + k2 * x2 + b

# Градиентный спуск
for epoch in range(100):
    for i in range(n_samples):
        target = y_values[i]
        output = f(x1_values[i], x2_values[i])
        error = output - target
        b -= lr * (2 / n_samples) * error
        k1 -= lr * (2 / n_samples) * error * x1_values[i]
        k2 -= lr * (2 / n_samples) * error * x2_values[i]

print(f"Final coefficients: k1={k1:.3f}, k2={k2:.3f}, b={b:.3f}")

# Визуализация
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Отображение исходных точек
ax.scatter(x1_values, x2_values, y_values, color="blue", label="Исходные данные", s=50)

# Построение плоскости регрессии
x1_grid = np.linspace(min(x1_values), max(x1_values), 20)
x2_grid = np.linspace(min(x2_values), max(x2_values), 20)
x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
y_pred = f(x1_mesh, x2_mesh)
ax.plot_surface(x1_mesh, x2_mesh, y_pred, alpha=0.5, color="red", label="Плоскость регрессии")

# Настройка осей и подписей
ax.set_xlabel("Площадь квартиры (м²)", fontsize=17)
ax.set_ylabel("Расстояние до центра (км)", fontsize=17)
ax.set_zlabel("Стоимость (тыс. $)", fontsize=17)
ax.legend()

plt.tight_layout()
plt.show()

