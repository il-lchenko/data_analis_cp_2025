import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# Создаем директории для результатов
os.makedirs('results/images', exist_ok=True)

# Загрузка данных
df = pd.read_csv('data/processed_IMDB.csv')

# Анализ длины отзывов
df['review_length'] = df['lemmatized'].apply(lambda x: len(str(x).split()))
print("Статистика длины отзывов:")
print(df['review_length'].describe())

# Визуализация распределения длины
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='review_length', bins=50, kde=True)
plt.title('Распределение длины отзывов')
plt.xlabel('Количество слов')
plt.ylabel('Частота')
plt.grid(True)
plt.savefig('results/images/review_length_distribution.png')
plt.close()

# Облака слов
def generate_wordcloud(text, filename):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(f'results/images/{filename}')
    plt.close()

positive_text = " ".join(df[df['sentiment'] == 'positive']['lemmatized'].astype(str))
negative_text = " ".join(df[df['sentiment'] == 'negative']['lemmatized'].astype(str))

generate_wordcloud(positive_text, 'wordcloud_positive.png')
generate_wordcloud(negative_text, 'wordcloud_negative.png')