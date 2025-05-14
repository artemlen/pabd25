import datetime
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import numpy as np
import glob
import cianparser
import logging
from typing import Optional
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

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
    """Парсит данные с cian.ru для указанного количества комнат."""
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
    """Очищает и подготавливает данные для обучения модели."""
    try:
        raw_data_path = './pabd25/data/raw'
        file_list = glob.glob(raw_data_path + "/*.csv")
        
        if not file_list:
            logger.warning("В директории raw нет CSV файлов для обработки")
            return pd.DataFrame()
        
        logger.info(f"Найдено {len(file_list)} CSV файлов для обработки")
        
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
        
        # Добавим новый признак - относительный этаж
        new_dataframe['relative_floor'] = new_dataframe['floor'] / new_dataframe['floors_count']
        
        # Сохранение очищенных данных
        cleaned_path = './pabd25/data/cleaned_data.csv'
        new_dataframe.to_csv(cleaned_path, index=False)
        logger.info(f"Очищенные данные сохранены в {cleaned_path}")
        
        return new_dataframe
    
    except Exception as e:
        logger.error(f"Ошибка при очистке данных: {str(e)}")
        raise

def train_and_evaluate_model(data: pd.DataFrame) -> GradientBoostingRegressor:
    """Обучает модель градиентного бустинга на подготовленных данных."""
    try:
        if data.empty:
            logger.error("Передан пустой DataFrame для обучения модели")
            raise ValueError("Пустой DataFrame")
        
        logger.info("Начало обучения модели")
        
        # Разделение на признаки и целевую переменную
        X = data[['total_meters', 'floor', 'floors_count', 'rooms_count', 'relative_floor']]
        y = data['price']
        
        # Логарифмирование цены для более нормального распределения
        y = np.log1p(y)
        
        # Разделение на тренировочную и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        logger.info(f"Данные разделены: train={len(X_train)}, test={len(X_test)}")
        
        # Создание пайплайна с масштабированием, отбором признаков и моделью
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_regression, k=4)),
            ('model', GradientBoostingRegressor(random_state=42))
        ])
        
        # Параметры для подбора
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 4, 5],
            'model__min_samples_split': [2, 5, 10],
            'model__subsample': [0.8, 0.9, 1.0]
        }
        
        # Поиск по сетке с кросс-валидацией
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        logger.info("Начало подбора гиперпараметров")
        grid_search.fit(X_train, y_train)
        logger.info("Подбор гиперпараметров завершен")
        
        # Лучшая модель
        model = grid_search.best_estimator_
        
        # Предсказание на тестовой выборке
        y_pred = model.predict(X_test)
        
        # Оценка модели (возвращаем к исходной шкале)
        y_test_exp = np.expm1(y_test)
        y_pred_exp = np.expm1(y_pred)
        
        mse = mean_squared_error(y_test_exp, y_pred_exp)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_exp, y_pred_exp)
        mae = mean_absolute_error(y_test_exp, y_pred_exp)
        
        logger.info(f"Лучшие параметры: {grid_search.best_params_}")
        logger.info(f"Метрики модели:\n"
                   f"MSE: {mse:.2f}\n"
                   f"RMSE: {rmse:.2f}\n"
                   f"R²: {r2:.6f}\n"
                   f"Средняя абсолютная ошибка: {mae:.2f} рублей")
        
        return model
    
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {str(e)}")
        raise

def save_model(model: GradientBoostingRegressor, model_path: str) -> None:
    """Сохраняет обученную модель в файл."""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Модель успешно сохранена в {model_path}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении модели: {str(e)}")
        raise

def load_model(model_path: str) -> Optional[GradientBoostingRegressor]:
    """Загружает модель из файла."""
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

def make_prediction(model: GradientBoostingRegressor, 
                   total_meters: float, 
                   floor: int, 
                   floors_count: int, 
                   rooms_count: int) -> float:
    """Делает предсказание цены квартиры с помощью модели."""
    try:
        relative_floor = floor / floors_count
        prediction = np.expm1(model.predict([[total_meters, floor, floors_count, rooms_count, relative_floor]])[0])
        logger.info(f"Предсказанная цена для квартиры {total_meters} M2, "
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
        for n_rooms in [1, 2, 3]:
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
        model_path = f'./pabd25/models/gb_house_price_model_{t}.pkl'
        save_model(model, model_path)
        
        # 5. Использование модели
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