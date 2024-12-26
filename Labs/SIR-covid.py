# Імпорт необхідних бібліотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from IPython.display import display

# 1. Завантаження даних
url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/jhu/full_data.csv'
data = pd.read_csv(url)

# Перегляд перших рядків даних
display(data.head())

# Видалення пропусків
data = data.dropna()
display(data.head())

# 2. Візуалізація даних
# Побудова графіків для Індії та Італії
data[data["location"] == "India"].plot()
data[data["location"] == "Italy"].plot()

# 3. Підготовка даних для моделювання
# Вибір країн
country_spike = "India"
country_decline = "Italy"

# Дані для Індії
india_data = data[data["location"] == country_spike]
india_data = india_data[["date", "total_cases", "total_deaths"]].fillna(0)
india_data["date"] = pd.to_datetime(india_data["date"])
india_data.set_index("date", inplace=True)

# Дані для Італії
italy_data = data[data["location"] == country_decline]
italy_data = italy_data[["date", "total_cases", "total_deaths"]].fillna(0)
italy_data["date"] = pd.to_datetime(italy_data["date"])
italy_data.set_index("date", inplace=True)

# Перевірка підготовлених даних
display(india_data.head())
display(italy_data.head())

# 4. Реалізація SIR-F моделі
def sirf_model(y, t, theta, kappa, rho, sigma):
    S, I, R, F = y
    N = max(S + I + R + F, 1e-10)  # Уникаємо ділення на нуль
    dS_dt = -theta * kappa * S * I / N
    dI_dt = theta * kappa * S * I / N - rho * I - sigma * I
    dR_dt = rho * I
    dF_dt = sigma * I
    return [dS_dt, dI_dt, dR_dt, dF_dt]

def fit_sirf_model(data, initial_conditions, t):
    def model(params):
        theta, kappa, rho, sigma = params
        solution = odeint(
            sirf_model,
            initial_conditions,
            t,
            args=(theta, kappa, rho, sigma),
            rtol=1e-6,
            atol=1e-8,
            mxstep=5000  # Максимальна кількість кроків
        )
        return solution[:, 1]  # Повертається лише кількість інфікованих

    I_actual = data["total_cases"].values
    params_opt, _ = curve_fit(
        lambda t, theta, kappa, rho, sigma: model([theta, kappa, rho, sigma]),
        t,
        I_actual,
        maxfev=5000,
        p0=[0.1, 0.1, 0.1, 0.01],
        bounds=([1e-5, 1e-5, 1e-5, 1e-5], [10, 10, 10, 10])
    )
    return params_opt



# 5. Початкові умови для Індії
N_india = 1_380_000_000  # Населення Індії
I0_india = india_data.iloc[0]["total_cases"]
F0_india = india_data.iloc[0]["total_deaths"]
R0_india = 0  # Початок без одужалих
S0_india = N_india - I0_india - R0_india - F0_india
initial_conditions_india = [S0_india, I0_india, R0_india, F0_india]
t_india = np.arange(len(india_data))

params_india = fit_sirf_model(india_data, initial_conditions_india, t_india)

# Початкові умови для Італії
N_italy = 60_360_000  # Населення Італії
I0_italy = italy_data.iloc[0]["total_cases"]
F0_italy = italy_data.iloc[0]["total_deaths"]
R0_italy = 0  # Початок без одужалих
S0_italy = N_italy - I0_italy - R0_italy - F0_italy
initial_conditions_italy = [S0_italy, I0_italy, R0_italy, F0_italy]
t_italy = np.arange(len(italy_data))

params_italy = fit_sirf_model(italy_data, initial_conditions_italy, t_italy)

# 6. Прогнозування на 300 днів
future_days = 300
t_future = np.arange(future_days)

sirf_india_future = odeint(sirf_model, initial_conditions_india, t_future, args=tuple(params_india))
sirf_italy_future = odeint(sirf_model, initial_conditions_italy, t_future, args=tuple(params_italy))

# 7. Візуалізація результатів
plt.figure(figsize=(14, 8))

# Інфіковані
plt.subplot(2, 2, 1)
plt.plot(t_future, sirf_india_future[:, 1], label="India: Infected", color="red")
plt.plot(t_future, sirf_italy_future[:, 1], label="Italy: Infected", color="orange")
plt.title("Infected (India vs Italy)")
plt.xlabel("Days")
plt.ylabel("Population")
plt.legend()
plt.grid()

# Одужалі
plt.subplot(2, 2, 2)
plt.plot(t_future, sirf_india_future[:, 2], label="India: Recovered", color="green")
plt.plot(t_future, sirf_italy_future[:, 2], label="Italy: Recovered", color="blue")
plt.title("Recovered (India vs Italy)")
plt.xlabel("Days")
plt.ylabel("Population")
plt.legend()
plt.grid()

# Летальні випадки
plt.subplot(2, 2, 3)
plt.plot(t_future, sirf_india_future[:, 3], label="India: Fatal", color="purple")
plt.plot(t_future, sirf_italy_future[:, 3], label="Italy: Fatal", color="black")
plt.title("Fatal (India vs Italy)")
plt.xlabel("Days")
plt.ylabel("Population")
plt.legend()
plt.grid()

# Вразливі
plt.subplot(2, 2, 4)
plt.plot(t_future, sirf_india_future[:, 0], label="India: Susceptible", color="cyan")
plt.plot(t_future, sirf_italy_future[:, 0], label="Italy: Susceptible", color="gray")
plt.title("Susceptible (India vs Italy)")
plt.xlabel("Days")
plt.ylabel("Population")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# 8. Виведення параметрів моделі
print(f"India - Theta: {params_india[0]:.4f}, Kappa: {params_india[1]:.4f}, Rho: {params_india[2]:.4f}, Sigma: {params_india[3]:.4f}")
print(f"Italy - Theta: {params_italy[0]:.4f}, Kappa: {params_italy[1]:.4f}, Rho: {params_italy[2]:.4f}, Sigma: {params_italy[3]:.4f}")
