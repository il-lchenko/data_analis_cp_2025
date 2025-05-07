import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
import os

# Настройка для корректного отображения графиков
plt.switch_backend('agg')  # Для сохранения файлов на сервере
# plt.switch_backend('TkAgg')  # Для отображения в IDE (раскомментировать при локальном запуске)

# Создаем директории для результатов
os.makedirs('results/images', exist_ok=True)

def bootstrap_analysis(data, n_iter=1000):
    """Вычисляет бутстрап-оценки среднего значения"""
    means = []
    for _ in range(n_iter):
        sample = resample(data)
        means.append(np.mean(sample))
    return means

def plot_bootstrap_results(pos_means, neg_means):
    """Визуализация результатов бутстрапа"""
    plt.figure(figsize=(12, 6))
    
    # Гистограммы распределений
    plt.hist(pos_means, bins=30, alpha=0.5, color='blue', label='Положительные отзывы')
    plt.hist(neg_means, bins=30, alpha=0.5, color='red', label='Отрицательные отзывы')
    
    # Линии для средних значений
    plt.axvline(np.mean(pos_means), color='blue', linestyle='--', linewidth=2)
    plt.axvline(np.mean(neg_means), color='red', linestyle='--', linewidth=2)
    
    plt.title('Распределение средней длины отзывов (бутстрап)', fontsize=14)
    plt.xlabel('Средняя длина отзыва (слов)', fontsize=12)
    plt.ylabel('Частота', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Сохраняем график
    plt.savefig('results/images/bootstrap_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()  # Закрываем фигуру чтобы освободить память

def main():
    # Загрузка данных
    try:
        df = pd.read_csv('data/processed_IMDB.csv')
        df['review_length'] = df['lemmatized'].apply(lambda x: len(str(x).split()))
    except FileNotFoundError:
        print("Ошибка: Файл data/processed_IMDB.csv не найден")
        return

    # Разделение на положительные и отрицательные отзывы
    positive = df[df['sentiment'] == 'positive']['review_length']
    negative = df[df['sentiment'] == 'negative']['review_length']

    # Бутстрап-анализ
    print("Выполнение бутстрап-анализа...")
    pos_means = bootstrap_analysis(positive)
    neg_means = bootstrap_analysis(negative)

    # Визуализация
    plot_bootstrap_results(pos_means, neg_means)
    print("График сохранен в results/images/bootstrap_distribution.png")

    # Доверительные интервалы
    def get_ci(means, confidence=95):
        lower = np.percentile(means, (100-confidence)/2)
        upper = np.percentile(means, 100 - (100-confidence)/2)
        return (lower, upper)

    pos_ci = get_ci(pos_means)
    neg_ci = get_ci(neg_means)

    print("\nРезультаты бутстрап-анализа:")
    print(f"Положительные отзывы 95% ДИ: ({pos_ci[0]:.2f}, {pos_ci[1]:.2f})")
    print(f"Отрицательные отзывы 95% ДИ: ({neg_ci[0]:.2f}, {neg_ci[1]:.2f})")

if __name__ == "__main__":
    main()