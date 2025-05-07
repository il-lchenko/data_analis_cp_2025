import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Настройки для подавления предупреждений
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Создаем директории для результатов
os.makedirs('results/images', exist_ok=True)

# Загрузка и подготовка данных
print("Загрузка данных...")
df = pd.read_csv('data/processed_IMDB.csv')
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    df['lemmatized'].astype(str),
    df['sentiment'],
    test_size=0.2,
    stratify=df['sentiment'],
    random_state=42
)

# Токенизация и паддинг
print("Токенизация текста...")
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = 200
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)

# Создание модели
print("Создание модели LSTM...")
model = Sequential([
    Embedding(20000, 128),  # Убрал input_length
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Обучение с callback'ами
print("Обучение модели...")
history = model.fit(
    X_train_pad,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test_pad, y_test),
    verbose=1
)

# Визуализация
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Обучение')
    plt.plot(history.history['val_accuracy'], label='Валидация')
    plt.title('Точность модели')
    plt.ylabel('Точность')
    plt.xlabel('Эпоха')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Обучение')
    plt.plot(history.history['val_loss'], label='Валидация')
    plt.title('Потери модели')
    plt.ylabel('Потери')
    plt.xlabel('Эпоха')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/images/training_history.png')
    plt.show()

plot_training_history(history)

# Оценка модели
print("\nФинальная оценка:")
test_loss, test_acc = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"Точность на тесте: {test_acc:.4f}")
print(f"Потери на тесте: {test_loss:.4f}")

# Сохранение модели
model.save('results/imdb_lstm_model.keras')
print("Модель сохранена в results/imdb_lstm_model.keras")