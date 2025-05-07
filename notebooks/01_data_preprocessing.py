# Импорт библиотек
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# Загрузка данных
df = pd.read_csv('data/processed_IMDB.csv')

# Анализ длины отзывов
df['review_length'] = df['lemmatized'].apply(lambda x: len(str(x).split()))
print("Статистика длины отзывов:")
print(df['review_length'].describe())

# Создание директории для изображений
os.makedirs('results/images', exist_ok=True)

# Визуализация распределения длины отзывов
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='review_length', bins=50, kde=True)
plt.title('Распределение длины отзывов')
plt.xlabel('Количество слов')
plt.ylabel('Частота')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/images/review_length_distribution.png')
plt.close()

# Облака слов
positive_text = " ".join(df[df['sentiment'] == 'positive']['lemmatized'].dropna())
negative_text = " ".join(df[df['sentiment'] == 'negative']['lemmatized'].dropna())

# Построение облаков слов
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

positive_wc = WordCloud(width=600, height=400, background_color='white').generate(positive_text)
negative_wc = WordCloud(width=600, height=400, background_color='white').generate(negative_text)

axes[0].imshow(positive_wc, interpolation='bilinear')
axes[0].set_title('Положительные отзывы')
axes[0].axis('off')

axes[1].imshow(negative_wc, interpolation='bilinear')
axes[1].set_title('Отрицательные отзывы')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('results/images/wordclouds.png')
plt.close()
