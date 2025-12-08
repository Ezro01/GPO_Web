"""
Скрипт для анализа датасета эмоций
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_dataset():
    """Анализирует датасет и выводит статистики"""
    print("=" * 60)
    print("АНАЛИЗ ДАТАСЕТА ЭМОЦИЙ")
    print("=" * 60)
    
    # Загрузка данных
    df = pd.read_csv('train.csv')
    
    print(f"\n📊 Общая статистика:")
    print(f"Всего записей: {len(df)}")
    print(f"Колонки: {', '.join(df.columns)}")
    
    print(f"\n📈 Распределение эмоций:")
    emotion_counts = df['emotion'].value_counts()
    print(emotion_counts)
    
    print(f"\n📊 Процентное распределение:")
    emotion_percent = df['emotion'].value_counts(normalize=True) * 100
    for emotion, percent in emotion_percent.items():
        print(f"{emotion:15s}: {percent:5.2f}%")
    
    # Проверка наличия файлов
    print(f"\n🔍 Проверка аудио файлов:")
    missing_files = []
    for idx, row in df.iterrows():
        file_path = os.path.join('train', row['path'])
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️  Найдено {len(missing_files)} отсутствующих файлов")
    else:
        print("✅ Все файлы найдены")
    
    # Статистика по длине текста
    print(f"\n📝 Статистика по тексту:")
    df['text_length'] = df['text'].str.len()
    print(f"Средняя длина текста: {df['text_length'].mean():.1f} символов")
    print(f"Мин. длина: {df['text_length'].min()} символов")
    print(f"Макс. длина: {df['text_length'].max()} символов")
    
    return df, emotion_counts

if __name__ == "__main__":
    df, emotion_counts = analyze_dataset()
    print("\n" + "=" * 60)
