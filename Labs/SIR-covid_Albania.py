import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from IPython.display import display

# 1. Завантаження даних
url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/jhu/full_data.csv'
data = pd.read_csv(url)


display(data.head())


data = data.dropna()
display(data.head())

data[data["location"] == "Albania"].plot()

country_analysis = "Albania"

albania_data = data[data["location"] == country_analysis]
albania_data = albania_data[["date", "total_cases", "total_deaths"]].fillna(0)
albania_data["date"] = pd.to_datetime(albania_data["date"])
albania_data.set_index("date", inplace=True)

# Перевірка підготовлених даних
display(albania_data.head())

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

# 5. Початкові умови для Албанії
N_albania = 2_850_000  # Населення Албанії
I0_albania = albania_data.iloc[0]["total_cases"]
F0_albania = albania_data.iloc[0]["total_deaths"]
R0_albania = 0  # Початок без одужалих
S0_albania = N_albania - I0_albania - R0_albania - F0_albania
initial_conditions_albania = [S0_albania, I0_albania, R0_albania, F0_albania]
t_albania = np.arange(len(albania_data))

params_albania = fit_sirf_model(albania_data, initial_conditions_albania, t_albania)

# 6. Прогнозування на 300 днів
future_days = 300
t_future = np.arange(future_days)

sirf_albania_future = odeint(sirf_model, initial_conditions_albania, t_future, args=tuple(params_albania))

# 7. Візуалізація результатів
plt.figure(figsize=(14, 8))

# Інфіковані
plt.subplot(2, 2, 1)
plt.plot(t_future, sirf_albania_future[:, 1], label="Albania: Infected", color="red")
plt.title("Infected (Albania)")
plt.xlabel("Days")
plt.ylabel("Population")
plt.legend()
plt.grid()

# Одужалі
plt.subplot(2, 2, 2)
plt.plot(t_future, sirf_albania_future[:, 2], label="Albania: Recovered", color="green")
plt.title("Recovered (Albania)")
plt.xlabel("Days")
plt.ylabel("Population")
plt.legend()
plt.grid()

# Летальні випадки
plt.subplot(2, 2, 3)
plt.plot(t_future, sirf_albania_future[:, 3], label="Albania: Fatal", color="purple")
plt.title("Fatal (Albania)")
plt.xlabel("Days")
plt.ylabel("Population")
plt.legend()
plt.grid()

# Вразливі
plt.subplot(2, 2, 4)
plt.plot(t_future, sirf_albania_future[:, 0], label="Albania: Susceptible", color="cyan")
plt.title("Susceptible (Albania)")
plt.xlabel("Days")
plt.ylabel("Population")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

print(f"Albania - Theta: {params_albania[0]:.4f}, Kappa: {params_albania[1]:.4f}, Rho: {params_albania[2]:.4f}, Sigma: {params_albania[3]:.4f}")
