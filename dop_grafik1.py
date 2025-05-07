import matplotlib.pyplot as plt
import pandas as pd

data = {'Метод': ['Логистическая регрессия', 'SVM', 'LSTM'],
        'Точность': [0.889, 0.882, 0.897],
        'Время обучения (мин)': [2.1, 3.4, 25.8],
        'Память (ГБ)': [1.2, 1.5, 4.3]}

df = pd.DataFrame(data)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.bar(df['Метод'], df['Точность'])
plt.title('Сравнение точности методов')
plt.ylabel('Accuracy')

plt.subplot(1,2,2)
plt.bar(df['Метод'], df['Время обучения (мин)'])
plt.title('Сравнение времени обучения')
plt.ylabel('Минуты')
plt.tight_layout()
plt.savefig('results/comparison_plot.png')