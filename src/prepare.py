# Импорт для prepare_data
from sklearn.model_selection import train_test_split
import datetime
import os
import pandas as pd
import logging

# Создание логера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/dvc.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def Prepare_data(output_dir='data/processed', input_dir='data/raw'):
    logger.info(f"Начало подготовки данных. Входная директория: {input_dir}, выходная директория: {output_dir}")
    
    # Получаю самый последний csv файл
    try:
        files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        if not files:
            logger.error(f"Во входной директории {input_dir} нет CSV файлов")
            raise FileNotFoundError(f"No CSV files in {input_dir}")
            
        latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(input_dir, x)))
        logger.info(f"Найден последний файл с данными: {latest_file}")
        
        df = pd.read_csv(input_dir+"/"+latest_file)
        logger.info(f"Файл успешно прочитан. Исходное количество строк: {len(df)}")
        
        df = df[["total_meters","rooms_count","floors_count","floor","price"]]
        logger.info("Выбраны только необходимые колонки")
        
        df = df.dropna()
        logger.info(f"Удалены строки с NaN. Осталось строк: {len(df)}")
        
        # Фильтрация данных
        initial_count = len(df)
        df = df[(df['price'] > 1000) & (df['price'] < 40000000)]
        df = df[(df['total_meters'] > 10) & (df['total_meters'] < 500)]
        filtered_count = len(df)
        logger.info(f"Применены фильтры. Удалено строк: {initial_count - filtered_count}. Осталось: {filtered_count}")
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['price']), df['price'], test_size=0.2, random_state=42)
        logger.info(f"Данные разделены на train/test. Train: {len(X_train)} строк, Test: {len(X_test)} строк")
        
        # Сохранение результатов
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Создана выходная директория {output_dir} (если не существовала)")
        
        csv_path_train = output_dir+"/"+"train.csv"
        csv_path_test = output_dir+"/"+"test.csv"
        
        X_train = pd.concat([X_train, y_train], ignore_index=True, axis=1)
        X_train.columns = ['total_meters', 'rooms_count', 'floors_count', 'floor', 'price']
        X_train.to_csv(csv_path_train, encoding='utf-8', index=False)
        logger.info(f"Train данные сохранены в {csv_path_train}")
        
        X_test = pd.concat([X_test, y_test], ignore_index=True, axis=1)
        X_test.columns = ['total_meters', 'rooms_count', 'floors_count', 'floor', 'price']
        X_test.to_csv(csv_path_test, encoding='utf-8', index=False)
        logger.info(f"Test данные сохранены в {csv_path_test}")
        
        logger.info("Подготовка данных успешно завершена")
        
    except Exception as e:
        logger.error(f"Ошибка при подготовке данных: {str(e)}")
        raise


Prepare_data()