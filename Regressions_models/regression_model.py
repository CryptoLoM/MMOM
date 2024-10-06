import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime, timedelta

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['logged_at'] = pd.to_datetime(df['logged_at'])
    return df

def filter_data_by_date(df):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    return df[(df['logged_at'] >= start_date) & (df['logged_at'] <= end_date)]

def preprocess_data(df):
    df.drop(columns=['device_id', 'value_text'], inplace=True)
    df.dropna(axis=0, subset=['value'], inplace=True)
    df = df.pivot_table(index='logged_at', columns='phenomenon', values='value')
    df.reset_index(inplace=True)
    return df

# Додаємо новий стовпець для розподілу на часові інтервали
def add_time_of_day(df):
    def get_time_of_day(hour):
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 24:
            return 'evening'
        else:
            return 'night'

    df['time_of_day'] = df['logged_at'].dt.hour.apply(get_time_of_day)
    return df

# Аналіз концентрації забрудників протягом доби
def analyze_by_time_of_day(df):
    # Групуємо дані за періодом дня та обчислюємо середнє значення для кожного забрудника
    time_of_day_avg = df.groupby('time_of_day')[['pm10', 'pm25']].mean()

    # Створюємо графік для візуалізації
    time_of_day_avg.plot(kind='bar', figsize=(10, 6), colormap='coolwarm')
    plt.title('Середня концентрація PM10 та PM2,5 за часом доби:')
    plt.ylabel('Концетрація(µg/m³):')
    plt.xlabel('Time of Day')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Кореляційна матриця
def correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[float, int])
    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(numeric_df.corr(method='pearson'), annot=True, cbar=False, cmap='coolwarm')
    heatmap.set_xlabel('')
    heatmap.set_ylabel('')
    plt.title('Correlation Matrix')
    plt.show()


# Графіки розсіювання
def scatter_plots(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 рядок та 2 стовпці підплотів
    if 'temperature' in df.columns and 'humidity' in df.columns:
        sns.scatterplot(x='temperature', y='humidity', data=df, s=1, ax=axes[1])
        axes[1].set_title('Temperature vs Humidity')

    if 'pm10' in df.columns and 'pm25' in df.columns:
        sns.scatterplot(x='pm10', y='pm25', data=df, s=1, ax=axes[0])
        axes[0].set_ylim(0, 100)
        axes[0].set_xlim(0, 150)
        axes[0].set_title('PM10 vs PM25')

    plt.tight_layout()
    plt.show()

# Аналіз залежностей PM10 і PM2.5
def analyze_pm(df):
    pm_df = df[["logged_at", "pm10", "pm25"]].dropna()  # Дропає NaN значення

    # Models for PM
    model_10_25 = LinearRegression()

    pm10_train, pm10_test, pm_10_25_train, pm_10_25_test = train_test_split(pm_df[['pm10']], pm_df[['pm25']], test_size=0.25, random_state=20, shuffle=True)

    # Train models
    model_10_25.fit(pm10_train, pm_10_25_train)
    pm_10_25_predicted = model_10_25.predict(pm10_test)

    # Visualization of PM models
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(x=pm10_train.values.flatten(), y=pm_10_25_train.values.flatten(), ax=ax, s=5, label='Train')
    sns.scatterplot(x=pm10_test.values.flatten(), y=pm_10_25_predicted.flatten(), ax=ax, s=5, label='Test', color='orange')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 150)
    ax.set_xlabel('PM10')
    ax.set_ylabel('PM25')
    ax.set_title('PM10 vs PM25')

    plt.legend()
    plt.show()

    print("R2 score for PM10/PM25: {:.2f}".format(r2_score(pm_10_25_test, pm_10_25_predicted)))
    print("RMSE score for PM10/PM25: {:.2f}".format(mean_squared_error(pm_10_25_test, pm_10_25_predicted, squared=False)))

# Основна функція
def main():
    file_path = 'saveecobot_23976.csv'
    df = load_data(file_path)
    df = filter_data_by_date(df)
    df = preprocess_data(df)

    print(df.head(15))

    # Додаємо новий стовпець з часовими інтервалами
    df = add_time_of_day(df)

    # Аналіз зміни забрудників протягом доби
    analyze_by_time_of_day(df)

    # Аналіз залежностей
    correlation_matrix(df)
    scatter_plots(df)
    analyze_pm(df)

    df = df.reset_index()
    df['day'] = df['logged_at'].dt.date
    print(df.head(15))

if __name__ == "__main__":
    main()

