from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

def evaluate_model(y_true, y_pred, model_name):
    """Оценка модели и сохранение результатов"""
    # Создаем директории, если их нет
    os.makedirs('results/images', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)
    
    try:
        # Метрики классификации
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'results/tables/{model_name}_metrics.csv', index=True)
        
        # Матрица ошибок
        plt.figure(figsize=(6,6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Матрица ошибок ({model_name})')
        plt.savefig(f'results/images/{model_name}_confusion_matrix.png')
        plt.close()
        
        return report_df
    except Exception as e:
        print(f"Ошибка в evaluate_model: {e}")
        raise