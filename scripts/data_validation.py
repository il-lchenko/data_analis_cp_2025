import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('results/images', exist_ok=True)

def validate_data(df):
    """Проверка качества данных"""
    # Проверка пропусков
    missing = df.isnull().sum()
    print("Missing values:\n", missing)
    
    # Визуализация пропусков
    plt.figure(figsize=(8,4))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.savefig('results/images/missing_values.png')
    plt.close()
    
    # Проверка дубликатов
    duplicates = df.duplicated().sum()
    print("\nNumber of duplicates:", duplicates)
    
    # Проверка баланса классов
    class_dist = df['sentiment'].value_counts()
    print("\nClass distribution:\n", class_dist)
    
    # Удаление дубликатов
    return df.drop_duplicates()

if __name__ == "__main__":
    data = pd.read_csv('../data/IMDB_Dataset.csv')
    cleaned_data = validate_data(data)
    cleaned_data.to_csv('../data/processed_IMDB.csv', index=False)