import numpy as np
import matplotlib.pyplot as plt


def is_inside_figure(x, y):
    # Визначення, чи знаходиться точка всередині фігури
    if (2 <= x <= 4 and 0 <= y <= 2) or (0 <= x <= 2 and 1 <= y <= 3):
        if not (0 <= x <= 2 and 0 <= y <= 1) or (2 <= x <= 4 and 2 <= y <= 3):
            return True
    return False


def monte_carlo(N=10000):
    x_random = np.random.uniform(0, 4, N)  # Генеруємо випадкові точки в межах прямокутника (0, 4) для x
    y_random = np.random.uniform(0, 4, N)  # Генеруємо випадкові точки в межах прямокутника (0, 4) для y
    rectangle_area = 4 * 4  # Площа прямокутника
    inside_count = 0

    for x, y in zip(x_random, y_random):
        if is_inside_figure(x, y):
            inside_count += 1

    figure_area = (inside_count / N) * rectangle_area
    return figure_area


N = 1000
calculated_area = monte_carlo(N)
print(f"Обчислена площа фігури методом Монте-Карло: {calculated_area:.4f}")

# Візуалізація
x = np.linspace(0, 4, 100)
y = np.linspace(0, 4, 100)
X, Y = np.meshgrid(x, y)
Z = np.vectorize(is_inside_figure)(X, Y)

plt.figure(figsize=(6, 6))
plt.imshow(Z, extent=(0, 4, 0, 4), origin='lower', cmap='Blues', alpha=0.6)
plt.title("Фігура для обчислення площі")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()
