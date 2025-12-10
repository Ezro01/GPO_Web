"""
Скрипт для создания изображения архитектуры модели с помощью matplotlib
"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os

def create_model_architecture_image(model_path='emotion_model.h5', 
                                   output_image='model_architecture.png'):
    """
    Создает визуальное представление архитектуры модели с помощью matplotlib
    
    Args:
        model_path: путь к файлу модели
        output_image: имя выходного файла изображения
    """
    
    print(f"🔍 Загрузка модели из: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Файл модели не найден: {model_path}")
        return
    
    try:
        # Загрузка модели
        model = keras.models.load_model(model_path)
        print(f"✅ Модель '{model.name}' успешно загружена")
        
        # Получаем информацию о слоях
        layers_info = []
        total_params = 0
        
        # Получаем summary в виде строки
        import io
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        summary_string = "\n".join(stringlist)
        
        # Собираем информацию о каждом слое
        for i, layer in enumerate(model.layers):
            layer_type = layer.__class__.__name__
            params = layer.count_params()
            total_params += params
            
            output_shape = layer.output_shape
            # Форматируем output shape для отображения
            if isinstance(output_shape, tuple):
                if len(output_shape) == 2:
                    out_shape_str = f"({output_shape[0]}, {output_shape[1]})"
                else:
                    out_shape_str = str(output_shape)
            else:
                out_shape_str = str(output_shape)
            
            activation = ""
            if hasattr(layer, 'activation'):
                if hasattr(layer.activation, '__name__'):
                    activation = layer.activation.__name__
            
            units = ""
            if hasattr(layer, 'units'):
                units = layer.units
            
            layers_info.append({
                'index': i,
                'name': layer.name,
                'type': layer_type,
                'params': params,
                'output_shape': out_shape_str,
                'activation': activation,
                'units': units
            })
        
        # Создаем график
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        
        # Левая часть: диаграмма слоев
        ax1.set_title(f"Архитектура модели: {model.name}", fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlim(0, 1)
        
        # Рассчитываем высоту для каждого слоя
        num_layers = len(layers_info)
        layer_height = 0.8 / num_layers
        
        # Рисуем слои
        for i, layer_info in enumerate(layers_info):
            y_pos = 0.9 - i * layer_height
            color = plt.cm.Set2(i / max(1, num_layers - 1))
            
            # Прямоугольник слоя
            rect = plt.Rectangle((0.2, y_pos - layer_height/2), 0.6, layer_height*0.8, 
                                fill=True, color=color, alpha=0.7, linewidth=2, edgecolor='darkblue')
            ax1.add_patch(rect)
            
            # Текст слоя
            layer_text = f"{layer_info['type']}\n"
            if layer_info['units']:
                layer_text += f"Нейроны: {layer_info['units']}\n"
            if layer_info['activation']:
                layer_text += f"Активация: {layer_info['activation']}\n"
            layer_text += f"Параметры: {layer_info['params']:,}"
            
            ax1.text(0.5, y_pos, layer_text, 
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Стрелки между слоями
            if i < num_layers - 1:
                next_y_pos = 0.9 - (i + 1) * layer_height
                ax1.arrow(0.5, y_pos - layer_height/2, 0, 
                         next_y_pos + layer_height/2 - (y_pos - layer_height/2),
                         head_width=0.02, head_length=0.02, fc='gray', ec='gray', alpha=0.7)
        
        # Настройка левой части
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.text(0.5, 0.95, f"Вход: {model.input_shape}", 
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
        
        output_text = f"Выход: {model.output_shape}\n"
        output_text += f"Классы: {model.output_shape[1]}"
        ax1.text(0.5, 0.02, output_text, 
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.8))
        
        # Правая часть: текстовая информация
        ax2.axis('off')
        ax2.set_title("Детали архитектуры", fontsize=16, fontweight='bold', pad=20)
        
        # Отображаем summary
        summary_text = f"Всего параметров: {total_params:,}\n"
        summary_text += f"Обучаемых параметров: {total_params:,}\n"
        summary_text += f"Необучаемых параметров: 0\n"
        summary_text += f"Количество слоев: {num_layers}\n"
        summary_text += "-" * 40 + "\n\n"
        
        # Добавляем информацию о каждом слое в виде таблицы
        table_data = []
        table_data.append(["Слой", "Тип", "Выход", "Параметры", "Активация"])
        table_data.append(["-"*10, "-"*10, "-"*10, "-"*10, "-"*10])
        
        for layer_info in layers_info:
            table_data.append([
                layer_info['name'],
                layer_info['type'],
                layer_info['output_shape'],
                f"{layer_info['params']:,}",
                layer_info['activation']
            ])
        
        # Создаем таблицу
        table = ax2.table(cellText=table_data, 
                         cellLoc='center', 
                         loc='center',
                         colWidths=[0.15, 0.2, 0.25, 0.15, 0.15])
        
        # Стилизуем таблицу
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Цвет для заголовков таблицы
        for i in range(2):
            for j in range(5):
                table[(i, j)].set_facecolor('#4C72B0')
                table[(i, j)].set_text_props(color='white', fontweight='bold')
        
        # Цвет для строк с данными
        for i in range(2, len(table_data)):
            color = '#F5F5F5' if i % 2 == 0 else '#E8E8E8'
            for j in range(5):
                table[(i, j)].set_facecolor(color)
        
        # Информация о модели
        info_text = f"\n\nИнформация о модели:\n"
        info_text += f"Имя модели: {model.name}\n"
        info_text += f"Входная форма: {model.input_shape}\n"
        info_text += f"Выходная форма: {model.output_shape}\n"
        info_text += f"Функция потерь: {model.loss}\n"
        
        if hasattr(model, 'optimizer'):
            optimizer_name = model.optimizer.__class__.__name__
            info_text += f"Оптимизатор: {optimizer_name}\n"
        
        ax2.text(0.5, 0.02, info_text, 
                ha='center', va='bottom', fontsize=10,
                transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
        
        # Настройка общего вида
        plt.suptitle(f"Модель классификации эмоций", fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Сохраняем изображение
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"✅ Изображение сохранено: {output_image}")
        print(f"📊 Статистика модели:")
        print(f"   - Количество слоев: {num_layers}")
        print(f"   - Всего параметров: {total_params:,}")
        print(f"   - Входная форма: {model.input_shape}")
        print(f"   - Выходная форма: {model.output_shape}")
        
        # Показываем изображение
        plt.show()
        
    except Exception as e:
        print(f"❌ Произошла ошибка: {str(e)}")

def create_simple_architecture(model_path='emotion_model.h5', 
                              output_image='model_simple.png'):
    """
    Создает упрощенное представление архитектуры
    """
    if not os.path.exists(model_path):
        print(f"❌ Файл модели не найден: {model_path}")
        return
    
    try:
        model = keras.models.load_model(model_path)
        
        # Получаем информацию о слоях
        layers = []
        for layer in model.layers:
            layer_info = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'params': layer.count_params()
            }
            if hasattr(layer, 'units'):
                layer_info['units'] = layer.units
            if hasattr(layer, 'activation'):
                if hasattr(layer.activation, '__name__'):
                    layer_info['activation'] = layer.activation.__name__
            layers.append(layer_info)
        
        # Создаем простой график
        fig, ax = plt.subplots(figsize=(10, len(layers) * 0.8 + 2))
        
        # Рисуем каждый слой
        for i, layer in enumerate(layers):
            y_pos = len(layers) - i - 1
            
            # Блок слоя
            color = plt.cm.coolwarm(i / max(1, len(layers) - 1))
            rect = plt.Rectangle((0.1, y_pos + 0.1), 0.8, 0.8, 
                                fill=True, color=color, alpha=0.7, 
                                edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Текст слоя
            text = f"{layer['type']}\n"
            if 'units' in layer:
                text += f"{layer['units']} нейронов\n"
            text += f"{layer['params']:,} параметров"
            
            ax.text(0.5, y_pos + 0.5, text, 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
            
            # Стрелка к следующему слою
            if i < len(layers) - 1:
                next_y_pos = len(layers) - (i + 1) - 1
                ax.arrow(0.5, y_pos + 0.1, 0, next_y_pos + 0.8 - (y_pos + 0.1),
                        head_width=0.03, head_length=0.05, fc='black', ec='black')
        
        # Настройки графика
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(layers) + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Заголовок
        ax.set_title(f"Архитектура модели\n{model.name}", 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Информация о входе и выходе
        ax.text(0.5, len(layers) + 0.3, 
               f"Вход: {model.input_shape} → {model.input_shape[1]} признаков", 
               ha='center', va='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen'))
        
        ax.text(0.5, -0.3, 
               f"Выход: {model.output_shape} → {model.output_shape[1]} классов", 
               ha='center', va='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral'))
        
        # Статистика
        total_params = model.count_params()
        ax.text(0.1, -0.3, 
               f"Всего параметров: {total_params:,}\nСлоев: {len(layers)}", 
               ha='left', va='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow'))
        
        plt.tight_layout()
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"✅ Упрощенная архитектура сохранена: {output_image}")
        plt.show()
        
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")

def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Создание изображения архитектуры модели с помощью matplotlib',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', type=str, default='emotion_model.h5',
                       help='Путь к файлу модели (по умолчанию: emotion_model.h5)')
    parser.add_argument('--output', type=str, default='model_architecture.png',
                       help='Имя выходного файла изображения')
    parser.add_argument('--simple', action='store_true',
                       help='Создать упрощенную версию архитектуры')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎨 СОЗДАНИЕ ИЗОБРАЖЕНИЯ АРХИТЕКТУРЫ МОДЕЛИ")
    print("="*60)
    
    if args.simple:
        print("\nСоздание упрощенной архитектуры...")
        create_simple_architecture(args.model, args.output)
    else:
        print("\nСоздание детализированной архитектуры...")
        create_model_architecture_image(args.model, args.output)
    
    print("\n" + "="*60)
    print("✅ ИЗОБРАЖЕНИЕ СОЗДАНО УСПЕШНО!")
    print("="*60)

if __name__ == "__main__":
    main()