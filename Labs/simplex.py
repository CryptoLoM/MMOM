import numpy as np


# Функція для виконання симплекс-методу
def simplex(v_coef, matrix, v_cons):
    m, n = matrix.shape
    # Створюємо таблицю симплекс-методу
    # Додаємо в таблицю коефіцієнти обмежень і цільову функцію
    table = np.hstack([matrix, np.eye(m), v_cons.reshape(-1, 1)])
    table = np.vstack([table, np.hstack([v_coef, np.zeros(m + 1)])])

    while True:
        # Перевірка умови оптимальності
        if all(table[-1, :-1] >= 0):
            # Якщо всі коефіцієнти в останньому ряду (окрім останньої колонки) позитивні або нульові
            return table[-1, -1], table[:-1, -1]  # Повертаємо оптимальне значення та змінні

        # Крок 1: Вибираємо змінну, що входить в базис
        # Вибираємо стовпець з мінімальним значенням в останньому ряду
        enter_column = np.argmin(table[-1, :-1])

        # Крок 2: Вибираємо змінну, що виходить з базису
        # Розраховуємо відношення правої частини до коефіцієнтів у вибраному стовпці
        ratios = table[:-1, -1] / table[:-1, enter_column]
        ratios[ratios <= 0] = np.inf  # Виключаємо від'ємні відношення
        leave_row = np.argmin(ratios)

        # Крок 3: Виконуємо операцію елементарного перетворення
        pivot = table[leave_row, enter_column]
        table[leave_row] /= pivot  # Ділимо ряд на елемент таблиці

        # Оновлюємо інші рядки таблиці
        for i in range(m + 1):
            if i != leave_row:
                table[i] -= table[i, enter_column] * table[leave_row]


# Приклад задачі лінійного програмування
# Коефіцієнти цільової функції
c = np.array([-2, -3, 0, 1, 0, 0])

# Коефіцієнти обмежень (ліва частина)
A = np.array([
    [2, -1, 0, -2, 1, 0],
    [3, 2, 1, -3, 0, 0],
    [-1, 3, 0, 4, 0, 1]
])

# Вільні члени обмежень (права частина)
b = np.array([16, 18, 24])

# Викликаємо симплекс-метод
optimal_value, solution = simplex(c, A, b)

# Виводимо результат
print("Оптимальні значення змінних:", solution)
print("Максимальне значення цільової функції:", optimal_value)