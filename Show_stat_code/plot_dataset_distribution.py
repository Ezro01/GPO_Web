"""
Скрипт для построения графика распределения классов эмоций в датасетах train и test
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Перевод эмоций на русский
emotion_translation = {
    'anger': 'Злость',
    'disgust': 'Отвращение',
    'enthusiasm': 'Энтузиазм',
    'fear': 'Страх',
    'happiness': 'Радость',
    'neutral': 'Нейтрально',
    'sadness': 'Грусть'
}

def plot_dataset_distribution():
    """Строит график распределения классов в train и test датасетах"""
    
    # Загрузка данных
    print("Загрузка данных...")
    train_df = pd.read_csv('/Users/roman_zverkov/Все папки по жизни/Универ/7 Семестр/ГПО/train.csv')
    test_df = pd.read_csv('/Users/roman_zverkov/Все папки по жизни/Универ/7 Семестр/ГПО/test.csv')
    
    # Подсчет количества по классам
    train_counts = train_df['emotion'].value_counts().sort_index()
    test_counts = test_df['emotion'].value_counts().sort_index()
    
    # Получаем все уникальные классы из обоих датасетов
    all_classes = sorted(set(train_counts.index) | set(test_counts.index))
    
    # Создание массивов значений для каждого класса
    train_values = [train_counts.get(cls, 0) for cls in all_classes]
    test_values = [test_counts.get(cls, 0) for cls in all_classes]
    total_values = [train_values[i] + test_values[i] for i in range(len(all_classes))]
    
    # Перевод названий классов на русский
    classes_ru = [emotion_translation.get(cls, cls) for cls in all_classes]
    
    # Создание графика
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Позиции для столбцов
    x = np.arange(len(all_classes))
    width = 0.25
    
    # Построение столбцов
    bars1 = ax.bar(x - width, train_values, width, label='Train', alpha=0.8, color='#4CAF50')
    bars2 = ax.bar(x, test_values, width, label='Test', alpha=0.8, color='#2196F3')
    bars3 = ax.bar(x + width, total_values, width, label='Всего (Train + Test)', alpha=0.8, color='#FF9800')
    
    # Добавление значений на столбцы
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # Настройка графика
    ax.set_xlabel('Эмоция', fontsize=14, fontweight='bold')
    ax.set_ylabel('Количество экземпляров', fontsize=14, fontweight='bold')
    ax.set_title('Russian Emotional Speech Dialogs', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes_ru, fontsize=12)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Поворот подписей оси X для лучшей читаемости
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    
    # Сохранение графика
    output_file = 'dataset_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ График сохранен: {output_file}")
    
    # Вывод статистики
    print("\n" + "="*60)
    print("СТАТИСТИКА ПО КЛАССАМ")
    print("="*60)
    print(f"\n{'Эмоция':<20} {'Train':<10} {'Test':<10} {'Всего':<10}")
    print("-"*60)
    
    for i, cls in enumerate(all_classes):
        train_val = train_values[i]
        test_val = test_values[i]
        total_val = total_values[i]
        cls_ru = classes_ru[i]
        print(f"{cls_ru:<20} {train_val:<10} {test_val:<10} {total_val:<10}")
    
    print("-"*60)
    print(f"{'ИТОГО':<20} {sum(train_values):<10} {sum(test_values):<10} {sum(total_values):<10}")
    
    # Процентное распределение
    print("\n" + "="*60)
    print("ПРОЦЕНТНОЕ РАСПРЕДЕЛЕНИЕ (от общего количества)")
    print("="*60)
    total_all = sum(total_values)
    for i, cls in enumerate(all_classes):
        cls_ru = classes_ru[i]
        percent = (total_values[i] / total_all) * 100
        print(f"{cls_ru:<20} {percent:>6.2f}%")
    
    plt.show()

if __name__ == "__main__":
    plot_dataset_distribution()

