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
            
            # Получаем параметр dropout если есть
            dropout_rate = ""
            if layer_type == 'Dropout':
                if hasattr(layer, 'rate'):
                    dropout_rate = f"rate={layer.rate}"
                elif hasattr(layer, 'get_config'):
                    config = layer.get_config()
                    if 'rate' in config:
                        dropout_rate = f"rate={config['rate']}"
            
            layers_info.append({
                'index': i,
                'name': layer.name,
                'type': layer_type,
                'params': params,
                'output_shape': out_shape_str,
                'activation': activation,
                'units': units,
                'dropout_rate': dropout_rate
            })
        
        # Создаем график
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        
        # Левая часть: диаграмма слоев
        ax1.set_title(f"Архитектура модели: {model.name}", fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlim(0, 1)
        
        # Рассчитываем высоту для каждого слоя
        num_layers = len(layers_info)
        layer_height = 0.8 / max(num_layers, 1)
        
        # Рисуем слои
        for i, layer_info in enumerate(layers_info):
            y_pos = 0.9 - i * layer_height
            color = plt.cm.Set2(i / max(1, num_layers - 1))
            
            # Определяем цвет для разных типов слоев
            if layer_info['type'] == 'Dense':
                color = '#4CAF50'  # Зеленый для Dense
            elif layer_info['type'] == 'Dropout':
                color = '#FF9800'  # Оранжевый для Dropout
            else:
                color = '#2196F3'  # Синий для других слоев
            
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
            if layer_info['dropout_rate']:
                layer_text += f"Dropout: {layer_info['dropout_rate']}\n"
            if layer_info['params'] > 0:
                layer_text += f"Параметры: {layer_info['params']:,}"
            else:
                layer_text += f"Параметры: 0"
            
            ax1.text(0.5, y_pos, layer_text, 
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
            
            # Стрелки между слоями
            if i < num_layers - 1:
                next_y_pos = 0.9 - (i + 1) * layer_height
                ax1.arrow(0.5, y_pos - layer_height/2, 0, 
                         next_y_pos + layer_height/2 - (y_pos - layer_height/2),
                         head_width=0.02, head_length=0.02, fc='gray', ec='gray', alpha=0.7, linewidth=1)
        
        # Настройка левой части
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # Вход
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        ax1.text(0.5, 0.95, f"Вход: {input_shape[1]} признаков", 
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.9))
        
        # Выход
        output_shape = model.output_shape
        num_classes = output_shape[1] if len(output_shape) > 1 else output_shape[0]
        output_text = f"Выход: {num_classes} классов\n(softmax)"
        
        ax1.text(0.5, 0.02, output_text, 
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.9))
        
        # Легенда
        legend_elements = [
            plt.Rectangle((0,0),1,1, fc='#4CAF50', alpha=0.7, label='Dense слои'),
            plt.Rectangle((0,0),1,1, fc='#FF9800', alpha=0.7, label='Dropout слои'),
            plt.Rectangle((0,0),1,1, fc='#2196F3', alpha=0.7, label='Другие слои'),
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
        
        # Правая часть: текстовая информация
        ax2.axis('off')
        ax2.set_title("Детали архитектуры", fontsize=16, fontweight='bold', pad=20)
        
        # Статистика модели
        trainable_params = np.sum([layer.count_params() for layer in model.layers if layer.trainable])
        non_trainable_params = total_params - trainable_params
        
        stats_text = f"📊 СТАТИСТИКА МОДЕЛИ\n"
        stats_text += "-" * 40 + "\n"
        stats_text += f"Всего параметров: {total_params:,}\n"
        stats_text += f"Обучаемых параметров: {trainable_params:,}\n"
        stats_text += f"Необучаемых параметров: {non_trainable_params:,}\n"
        stats_text += f"Количество слоев: {num_layers}\n\n"
        
        # Информация о dropout слоях
        dropout_layers = [l for l in layers_info if l['type'] == 'Dropout']
        if dropout_layers:
            stats_text += f"🔽 DROPOUT СЛОИ:\n"
            for dl in dropout_layers:
                stats_text += f"  • {dl['name']}: {dl['dropout_rate']}\n"
            stats_text += "\n"
        
        # Информация о dense слоях
        dense_layers = [l for l in layers_info if l['type'] == 'Dense']
        if dense_layers:
            stats_text += f"🧮 DENSE СЛОИ:\n"
            for dl in dense_layers:
                stats_text += f"  • {dl['name']}: {dl['units']} нейронов, {dl['activation']}\n"
            stats_text += "\n"
        
        # Информация о конфигурации обучения
        stats_text += f"⚙️ КОНФИГУРАЦИЯ ОБУЧЕНИЯ:\n"
        stats_text += f"  Функция потерь: {model.loss}\n"
        if hasattr(model, 'optimizer'):
            optimizer = model.optimizer
            stats_text += f"  Оптимизатор: {optimizer.__class__.__name__}\n"
            # Получаем learning rate
            try:
                lr = optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, 'numpy') else optimizer.learning_rate
                stats_text += f"  Learning rate: {lr}\n"
            except:
                pass
        
        ax2.text(0.02, 0.95, stats_text, 
                ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9),
                transform=ax2.transAxes)
        
        # Таблица с деталями слоев
        table_data = []
        table_data.append(["№", "Слой", "Тип", "Выход", "Параметры", "Детали"])
        table_data.append(["-"*3, "-"*10, "-"*10, "-"*12, "-"*10, "-"*15])
        
        for layer_info in layers_info:
            details = ""
            if layer_info['units']:
                details += f"{layer_info['units']} нейронов, "
            if layer_info['activation']:
                details += f"активация: {layer_info['activation']}, "
            if layer_info['dropout_rate']:
                details += f"{layer_info['dropout_rate']}"
            details = details.rstrip(", ")
            
            table_data.append([
                layer_info['index'] + 1,
                layer_info['name'][:10] + ("..." if len(layer_info['name']) > 10 else ""),
                layer_info['type'],
                layer_info['output_shape'][:12] + ("..." if len(layer_info['output_shape']) > 12 else ""),
                f"{layer_info['params']:,}",
                details[:20] + ("..." if len(details) > 20 else "")
            ])
        
        # Создаем таблицу
        table = ax2.table(cellText=table_data, 
                         cellLoc='center', 
                         loc='center',
                         colWidths=[0.05, 0.12, 0.12, 0.15, 0.12, 0.2],
                         bbox=[0.02, 0.02, 0.96, 0.5])
        
        # Стилизуем таблицу
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        
        # Цвет для заголовков таблицы
        for i in range(2):
            for j in range(6):
                table[(i, j)].set_facecolor('#4C72B0')
                table[(i, j)].set_text_props(color='white', fontweight='bold')
        
        # Цвет для строк с данными
        for i in range(2, len(table_data)):
            # Разные цвета для разных типов слоев
            cell_type = table_data[i][2]
            if cell_type == 'Dense':
                color = '#E8F5E8'
            elif cell_type == 'Dropout':
                color = '#FFF3E0'
            else:
                color = '#E8EAF6' if i % 2 == 0 else '#F5F5F5'
            
            for j in range(6):
                table[(i, j)].set_facecolor(color)
                table[(i, j)].set_edgecolor('#DDDDDD')
        
        # Настройка общего вида
        plt.suptitle(f"Модель классификации эмоций - Полная архитектура", 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Информация о параметрах dropout
        dropout_summary = "📉 СВОДКА ПО DROPOUT: "
        if dropout_layers:
            rates = []
            for dl in dropout_layers:
                if dl['dropout_rate']:
                    try:
                        rate = float(dl['dropout_rate'].split('=')[1])
                        rates.append(rate)
                    except:
                        pass
            
            if rates:
                dropout_summary += f"Всего {len(dropout_layers)} dropout слоев: "
                dropout_summary += ", ".join([f"Dropout({r})" for r in rates])
            else:
                dropout_summary += f"Всего {len(dropout_layers)} dropout слоев"
        else:
            dropout_summary += "Нет dropout слоев"
        
        plt.figtext(0.5, 0.01, dropout_summary, 
                   ha='center', va='bottom', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='#E0F2F1', alpha=0.9))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Сохраняем изображение
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"✅ Изображение сохранено: {output_image}")
        print(f"\n📊 ДЕТАЛЬНАЯ СТАТИСТИКА МОДЕЛИ:")
        print(f"   - Всего слоев: {num_layers}")
        print(f"   - Всего параметров: {total_params:,}")
        print(f"   - Обучаемых параметров: {trainable_params:,}")
        print(f"   - Входная форма: {input_shape}")
        print(f"   - Выходная форма: {output_shape}")
        print(f"   - Количество классов: {num_classes}")
        
        if dropout_layers:
            print(f"\n🔽 DROPOUT СЛОИ:")
            for dl in dropout_layers:
                print(f"   - {dl['name']}: {dl['dropout_rate']}")
        
        print(f"\n🧮 DENSE СЛОИ:")
        for dl in dense_layers:
            print(f"   - {dl['name']}: {dl['units']} нейронов, активация: {dl['activation']}")
        
        # Показываем изображение
        plt.show()
        
    except Exception as e:
        print(f"❌ Произошла ошибка: {str(e)}")
        import traceback
        traceback.print_exc()

def create_simple_architecture(model_path='emotion_model.h5', 
                              output_image='model_simple.png'):
    """
    Создает упрощенное представление архитектуры с отображением dropout
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
            
            # Получаем dropout rate
            if layer.__class__.__name__ == 'Dropout':
                if hasattr(layer, 'rate'):
                    layer_info['dropout_rate'] = layer.rate
                elif hasattr(layer, 'get_config'):
                    config = layer.get_config()
                    if 'rate' in config:
                        layer_info['dropout_rate'] = config['rate']
            
            layers.append(layer_info)
        
        # Создаем простой график
        fig, ax = plt.subplots(figsize=(12, len(layers) * 0.8 + 2))
        
        # Рисуем каждый слой
        for i, layer in enumerate(layers):
            y_pos = len(layers) - i - 1
            
            # Определяем цвет слоя
            if layer['type'] == 'Dense':
                color = '#4CAF50'  # Зеленый
            elif layer['type'] == 'Dropout':
                color = '#FF9800'  # Оранжевый
            else:
                color = '#2196F3'  # Синий
            
            # Блок слоя
            rect = plt.Rectangle((0.1, y_pos + 0.1), 0.8, 0.8, 
                                fill=True, color=color, alpha=0.7, 
                                edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Текст слоя
            text = f"{layer['type']}\n"
            if 'units' in layer:
                text += f"{layer['units']} нейронов\n"
            if 'activation' in layer and layer['activation']:
                text += f"активация: {layer['activation']}\n"
            if 'dropout_rate' in layer:
                text += f"Dropout: {layer['dropout_rate']}\n"
            text += f"{layer['params']:,} параметров"
            
            ax.text(0.5, y_pos + 0.5, text, 
                    ha='center', va='center', fontsize=9, fontweight='bold',
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
        
        # Статистика dropout
        dropout_layers = [l for l in layers if l['type'] == 'Dropout']
        if dropout_layers:
            dropout_info = "Dropout слои: "
            rates = []
            for dl in dropout_layers:
                if 'dropout_rate' in dl:
                    rates.append(str(dl['dropout_rate']))
            if rates:
                dropout_info += ", ".join([f"Dropout({r})" for r in rates])
            else:
                dropout_info += f"{len(dropout_layers)} слоев"
            
            ax.text(0.5, -0.2, dropout_info, 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='#FFF3E0'))
        
        # Информация о входе и выходе
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        ax.text(0.5, len(layers) + 0.3, 
               f"Вход: {input_shape[1]} признаков", 
               ha='center', va='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen'))
        
        output_shape = model.output_shape
        num_classes = output_shape[1] if len(output_shape) > 1 else output_shape[0]
        ax.text(0.5, -0.4, 
               f"Выход: {num_classes} классов", 
               ha='center', va='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral'))
        
        # Статистика
        total_params = model.count_params()
        ax.text(0.1, -0.4, 
               f"Всего параметров: {total_params:,}\nСлоев: {len(layers)}", 
               ha='left', va='center', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow'))
        
        plt.tight_layout()
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"✅ Упрощенная архитектура сохранена: {output_image}")
        
        # Вывод информации о dropout в консоль
        if dropout_layers:
            print(f"\n🔽 DROPOUT СЛОИ В МОДЕЛИ:")
            for dl in dropout_layers:
                rate_info = dl.get('dropout_rate', 'не указан')
                print(f"   - {dl['name']}: Dropout rate = {rate_info}")
        
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