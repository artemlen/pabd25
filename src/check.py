from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import datetime
import os
import pickle
import logging

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/dvc.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def Check_model(test_data, input_dir='models'):
    logger.info(f"Начало проверки модели. Входная директория: {input_dir}")
    logger.info(f"Получены входные данные: {test_data}")
    
    try:
        # Загрузка модели
        model_path = os.path.join(input_dir, "model.pkl")
        logger.info(f"Попытка загрузки модели из: {model_path}")
        
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)
            logger.info("Модель успешно загружена")
            
            # Подготовка данных
            column_names = ["total_meters", "rooms_count", "floors_count", "floor"]
            logger.info(f"Подготовка DataFrame с колонками: {column_names}")
            
            test_data = pd.DataFrame([test_data], columns=column_names)
            logger.debug(f"Сформирован DataFrame:\n{test_data}")
            
            # Предсказание
            logger.info("Выполнение предсказания...")
            predict = int(loaded_model.predict(test_data).round(0))
            logger.info(f"Получено предсказание: {predict}")
            
            # Вывод результата
            print(predict)
            logger.info("Предсказание успешно выведено")
            
    except FileNotFoundError:
        logger.error(f"Файл модели не найден по пути: {model_path}")
        raise
    except Exception as e:
        logger.error(f"Ошибка при выполнении предсказания: {str(e)}")
        raise
    finally:
        logger.info("Завершение проверки модели")

Check_model([50, 3, 17, 4])