import datetime
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import glob
import cianparser
import logging
from typing import Tuple, Optional

# Настройка логирования
def setup_logging(log_file: str = './pabd25/logs/app.log') -> None:
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Логирование настроено")

# Инициализация логгера
logger = logging.getLogger(__name__)

# Инициализация парсера
moscow_parser = cianparser.CianParser(location="Москва")

def parse_flats_data(n_rooms: int, output_dir: str = './pabd25/data/raw') -> pd.DataFrame:
    """
    Парсит данные с cian.ru для указанного количества комнат.
    
    Args:
        n_rooms: Количество комнат (1, 2 или 3)
        output_dir: Директория для сохранения данных
        
    Returns:
        DataFrame с данными о квартирах
    """
    try:
        t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        csv_path = f'{output_dir}/{n_rooms}к_{t}.csv'
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Начало парсинга данных для {n_rooms}-комнатных квартир")
        
        data = moscow_parser.get_flats(
            deal_type="sale",
            rooms=(n_rooms,),
            with_saving_csv=False,
            additional_settings={
                "start_page": 1,
                "end_page": 10,
                "object_type": "secondary"
            })
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, encoding='utf-8', index=False)
        logger.info(f"Сохранено {len(df)} записей для {n_rooms}-комнатных квартир в {csv_path}")
        return df
    
    except Exception as e:
        logger.error(f"Ошибка при парсинге данных для {n_rooms}-комнатных квартир: {str(e)}")
        raise

def clean_and_prepare_data() -> pd.DataFrame:
    """
    Очищает и подготавливает данные для обучения модели.
    
    Returns:
        Очищенный DataFrame
    """
    try:
        # Загрузка всех CSV файлов из директории
        raw_data_path = './pabd25/data/raw'
        file_list = glob.glob(raw_data_path + "/*.csv")
        
        if not file_list:
            logger.warning("В директории raw нет CSV файлов для обработки")
            return pd.DataFrame()
        
        logger.info(f"Найдено {len(file_list)} CSV файлов для обработки")
        
        # Объединение данных
        main_dataframe = pd.read_csv(file_list[0], delimiter=',')
        for i in range(1, len(file_list)):
            data = pd.read_csv(file_list[i], delimiter=',')
            df = pd.DataFrame(data)
            main_dataframe = pd.concat([main_dataframe, df], axis=0)
        
        initial_count = len(main_dataframe)
        logger.info(f"Объединенный датасет содержит {initial_count} записей")
        
        # Выбор нужных столбцов
        new_dataframe = main_dataframe[['total_meters', 'price', 'floor', 'floors_count', 'rooms_count']]
        
        # Удаление пропущенных значений
        new_dataframe = new_dataframe.dropna()
        logger.info(f"Удалено {initial_count - len(new_dataframe)} строк с пропущенными значениями")
        
        # Удаление дубликатов
        new_dataframe = new_dataframe.drop_duplicates()
        logger.info(f"Удалено {len(main_dataframe) - len(new_dataframe)} дубликатов")
        
        # Удаление выбросов
        def remove_outliers(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        initial_size = len(new_dataframe)
        for col in ['total_meters', 'price', 'floor', 'floors_count', 'rooms_count']:
            new_dataframe = remove_outliers(new_dataframe, col)
        
        logger.info(f"Удалено {initial_size - len(new_dataframe)} выбросов")
        logger.info(f"Итоговый размер датасета: {len(new_dataframe)} строк")
        
        # Сохранение очищенных данных
        cleaned_path = './pabd25/data/cleaned_data.csv'
        new_dataframe.to_csv(cleaned_path, index=False)
        logger.info(f"Очищенные данные сохранены в {cleaned_path}")
        
        return new_dataframe
    
    except Exception as e:
        logger.error(f"Ошибка при очистке данных: {str(e)}")
        raise

def train_and_evaluate_model(data: pd.DataFrame) -> LinearRegression:
    """
    Обучает модель линейной регрессии на подготовленных данных.
    
    Args:
        data: DataFrame с подготовленными данными
        
    Returns:
        Обученная модель LinearRegression
    """
    try:
        if data.empty:
            logger.error("Передан пустой DataFrame для обучения модели")
            raise ValueError("Пустой DataFrame")
        
        logger.info("Начало обучения модели")
        
        # Разделение на признаки и целевую переменную
        X = data[['total_meters', 'floor', 'floors_count', 'rooms_count']]
        y = data['price']
        
        # Разделение на тренировочную и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        logger.info(f"Данные разделены: train={len(X_train)}, test={len(X_test)}")
        
        # Обучение модели
        model = LinearRegression()
        model.fit(X_train, y_train)
        logger.info("Модель успешно обучена")
        
        # Предсказание на тестовой выборке
        y_pred = model.predict(X_test)
        
        # Оценка модели
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mean_abs_error = np.mean(np.abs(y_test - y_pred))
        
        # Логирование метрик
        logger.info(f"Метрики модели:\n"
                   f"MSE: {mse:.2f}\n"
                   f"RMSE: {rmse:.2f}\n"
                   f"R²: {r2:.6f}\n"
                   f"Средняя абсолютная ошибка: {mean_abs_error:.2f} рублей\n"
                   f"Коэффициенты: {model.coef_}\n"
                   f"Свободный член: {model.intercept_:.2f}")
        
        return model
    
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {str(e)}")
        raise

def save_model(model: LinearRegression, model_path: str) -> None:
    """
    Сохраняет обученную модель в файл.
    
    Args:
        model: Обученная модель
        model_path: Путь для сохранения модели
    """
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Модель успешно сохранена в {model_path}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении модели: {str(e)}")
        raise

def load_model(model_path: str) -> Optional[LinearRegression]:
    """
    Загружает модель из файла.
    
    Args:
        model_path: Путь к файлу с моделью
        
    Returns:
        Загруженная модель или None в случае ошибки
    """
    try:
        if not os.path.exists(model_path):
            logger.error(f"Файл модели не найден: {model_path}")
            return None
        
        model = joblib.load(model_path)
        logger.info(f"Модель успешно загружена из {model_path}")
        return model
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        raise

def make_prediction(model: LinearRegression, 
                   total_meters: float, 
                   floor: int, 
                   floors_count: int, 
                   rooms_count: int) -> float:
    """
    Делает предсказание цены квартиры с помощью модели.
    
    Args:
        model: Обученная модель
        total_meters: Площадь квартиры
        floor: Этаж
        floors_count: Всего этажей в доме
        rooms_count: Количество комнат
        
    Returns:
        Предсказанная цена квартиры
    """
    try:
        prediction = model.predict([[total_meters, floor, floors_count, rooms_count]])[0]
        logger.info(f"Предсказанная цена для квартиры {total_meters} м², "
                   f"{rooms_count} комнат, этаж {floor}/{floors_count}: {prediction:.2f} рублей")
        return prediction
    except Exception as e:
        logger.error(f"Ошибка при выполнении предсказания: {str(e)}")
        raise

def main():
    """Основная функция для выполнения всего пайплайна."""
    setup_logging()
    
    try:
        logger.info("Запуск пайплайна обработки данных")
        
        # 1. Парсинг данных
        logger.info("Начало этапа парсинга данных")
        for n_rooms in [1, 2]:
            parse_flats_data(n_rooms)
        
        # 2. Очистка и подготовка данных
        logger.info("Начало этапа очистки данных")
        cleaned_data = clean_and_prepare_data()
        
        if cleaned_data.empty:
            logger.error("Не удалось получить данные для обучения")
            return
        
        # 3. Обучение модели
        logger.info("Начало этапа обучения модели")
        model = train_and_evaluate_model(cleaned_data)
        
        # 4. Сохранение модели
        t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        model_path = f'./pabd25/models/house_price_model_{t}.pkl'
        save_model(model, model_path)
        
        # 5. Пример использования модели
        logger.info("Тестирование модели на примере")
        loaded_model = load_model(model_path)
        if loaded_model:
            make_prediction(loaded_model, 50, 5, 10, 2)
        
        logger.info("Пайплайн успешно завершен")
    
    except Exception as e:
        logger.critical(f"Критическая ошибка в пайплайне: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()