import sys
import os
from pathlib import Path

# Добавляем папку scripts в путь поиска модулей
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

try:
    from model_evaluation import evaluate_model
except ModuleNotFoundError:
    print("Ошибка: Не удалось загрузить модуль model_evaluation")
    print("Убедитесь, что файл model_evaluation.py находится в папке scripts/")
    sys.exit(1)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# Проверка и создание директорий
os.makedirs('results/images', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

# Загрузка данных
try:
    df = pd.read_csv('data/processed_IMDB.csv')
except FileNotFoundError:
    print("Ошибка: Файл data/processed_IMDB.csv не найден")
    print("Сначала выполните скрипты предварительной обработки данных")
    sys.exit(1)

# Преобразование меток
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    df['lemmatized'].astype(str),
    df['sentiment'],
    test_size=0.2,
    stratify=df['sentiment'],
    random_state=42
)

# Логистическая регрессия
print("\n" + "="*50)
print("Обучение модели логистической регрессии")
print("="*50)

lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

lr_pipeline.fit(X_train, y_train)
lr_predictions = lr_pipeline.predict(X_test)

print("\nОтчет о классификации:")
print(classification_report(y_test, lr_predictions))

try:
    evaluate_model(y_test, lr_predictions, 'logistic_regression')
    print("\nРезультаты сохранены в results/")
except Exception as e:
    print(f"\nОшибка при оценке модели: {e}")

# SVM
print("\n" + "="*50)
print("Обучение модели SVM")
print("="*50)

svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('classifier', LinearSVC())
])

svm_pipeline.fit(X_train, y_train)
svm_predictions = svm_pipeline.predict(X_test)

print("\nОтчет о классификации:")
print(classification_report(y_test, svm_predictions))

try:
    evaluate_model(y_test, svm_predictions, 'svm')
    print("\nРезультаты сохранены в results/")
except Exception as e:
    print(f"\nОшибка при оценке модели: {e}")