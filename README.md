## Housing Price Prediction Model

### Описание проекта
Проект направлен на создание модели машинного обучения для прогнозирования цен на жилье. Модель использует различные характеристики объектов недвижимости для предсказания их рыночной стоимости.

### Структура проекта
```
housing_price_prediction/
├── data/
│   ├── raw/                    # Исходные данные
├── logs/                       # Логи программ
├── models/                     # Обученные модели
├── notebooks/                  # Jupyter notebooks
├── servise/                    # Сервис предсказания цены на недвижимость
│   ├── templates/              # Шаблоны для веб-приложения
│   ├── app.py                  # Flask приложение
│   └── CreateModelLinearReg.py # Создание модели для предсказания цен
├── src/                        # Исходный код
├── requirements.txt            # Требования к зависимостям
├── start_commands.txt          # Команды для быстрого развертывания приложения 
└── README.md
```

### Архитектура сервиса ПА
![](img/arch.png)

### Данные
Используемые данные для обучения модели включают в себя следующие признаки:
* Площадь жилья (total_meters)
* Количество комнат (rooms_count)
* Количество этажей (floors_count)
* Номер этажа (floor)


В выборке 1, 2, 3, 4 комнатные квартиры 
* Всего 1476 записей
 


### Как запустить
1. Клонируйте репозиторий:
```bash
git clone https://github.com/artemlen/pabd25.git
```

2. Создайте venv и установите зависимости:
```bash
python -m venv venv
venv/Scripts/activate # Для Windows
source venv/bin/activate # Для Mac OS
pip install -r requirements.txt
```

3. Запустите цикл сбора данных и обучения:
```bash
python ./servise/CreateModelLinearReg.py
```
4. Запустите flusk приложение:
```bash
python ./pabd25/Servise/app.py
```

### Модели машинного обучения
* **Linear Regression** - Линейная регрессия


### Метрики оценки
* **Mean Squared Error (MSE)**
* **R² Score**

### Метрики модели
* MSE: 35528153034599.84
* R²: 0.4999135855413235

### Как использовать модель
1. Загрузите данные в формате CSV
2. Обработайте данные с помощью предобработчиков. В данных должно остаться 4 признака:
* total_meters
* rooms_count
* floors_count
* floor
3. Загрузите обученную модель из файла ".pkl"
```bash
with open(model_path, 'rb') as file:
    model = pickle.load(file)
```
4. Сделайте предсказания
```bash
prediction = int(model.predict(input_df).round(0))
```

### Использование сервиса предиктивной аналитики в dev mode
1. Запустите сервис с указанием имени модели
```sh
python ./pabd25/Servise/app.py
```
2. Веб приложение доступно по ссылке `http://127.0.0.1:8000` 
3. API endpoint доступен  по ссылке `http://127.0.0.1:8000/api/numbers`


### Автор
Артём Ленгауэр


### Контакты
* Email: artem_len@mail.ru