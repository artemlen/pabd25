# Импорт для Linear_model_train
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

def save_model(output_dir='models', name="linear"):
    logger.info(f"Начало обучения модели. Выходная директория: {output_dir}, имя модели: {name}")
    
    try:
        # Загрузка данных
        logger.info("Загрузка train и test данных...")
        train_df = pd.read_csv("./data/processed/train.csv")
        test_df = pd.read_csv("./data/processed/test.csv")
        logger.info(f"Данные загружены. Train: {len(train_df)} строк, Test: {len(test_df)} строк")
        
        # Объединение данных
        df = pd.concat([train_df, test_df], ignore_index=True)
        logger.info(f"Данные объединены. Всего строк: {len(df)}")
        
        # Разделение на признаки и целевую переменную
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(columns=['price']), 
            df['price'], 
            test_size=0.2, 
            random_state=42
        )
        logger.info(f"Данные разделены. X_train: {X_train.shape}, X_test: {X_test.shape}")
        
        # Обучение модели
        logger.info("Обучение модели LinearRegression...")
        modelLR = LinearRegression()
        modelLR.fit(X_train, y_train)
        logger.info("Модель успешно обучена")
        
        # Предсказания и метрики
        predictions = modelLR.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        logger.info(f"Метрики модели - MSE: {mse:.2f}, R2: {r2:.2f}")
        
        # Сохранение модели
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f'{name}_model.pkl')
        
        with open(model_path, 'wb') as file:
            pickle.dump(modelLR, file)
        logger.info(f"Модель сохранена по пути: {model_path}")
        
        logger.info("Процесс обучения и сохранения модели завершен успешно")
        
    except FileNotFoundError as e:
        logger.error(f"Ошибка: файл не найден - {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при обучении модели: {str(e)}")
        raise

save_model()