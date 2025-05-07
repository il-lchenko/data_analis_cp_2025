import re
import spacy
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')

def clean_text(text):
    """Очистка текста"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)  # Удаление HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Удаление спецсимволов
    return text.lower()

def lemmatize_text(text):
    """Лемматизация текста"""
    if not isinstance(text, str):
        return ""
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])

def preprocess_data(df, text_col='review'):
    """Предобработка данных"""
    tqdm.pandas()
    df['cleaned_text'] = df[text_col].progress_apply(clean_text)
    df['lemmatized'] = df['cleaned_text'].progress_apply(lemmatize_text)
    return df