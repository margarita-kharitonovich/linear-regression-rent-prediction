from random import randint, uniform
from matplotlib import pyplot as plt

n_samples = 30
x_values = [randint(28, 95) for _ in range(n_samples)]
y_values = [int(0.8 * x + uniform(-5, 5)) for x in x_values]
print(y_values)
lr = 0.001
b, k = randint(-100, 100) / 100, randint(-100, 100) / 100

def f(x):
    return k * x + b

def mse():
    errors = []
    for i in range(n_samples):
        target = y_values[i]
        output = f(x_values[i])
        errors.append((target - output) ** 2)
    return sum(errors) / n_samples

for epoch in range(100):
    for i in range(n_samples):
        target = y_values[i]
        output = f(x_values[i])
        b -= lr * (2 / n_samples) * (output - target)
        k -= lr * (2 / n_samples) * (output - target) * x_values[i]
    print(f"iteration: {epoch}, error: {mse()}")

print(k, b)

plt.scatter(x_values, y_values, label="Данные", color="blue")  # Точки данных
plt.plot(x_values, [f(x) for x in x_values], label="Линейная регрессия", color="red")  # Прямая
plt.xlabel("Площадь квартиры (м²)", fontsize=16)
plt.ylabel("Стоимость (тыс. $)", fontsize=16)
plt.legend()
plt.grid()
plt.show()