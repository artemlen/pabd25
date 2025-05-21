# Импорт для parse_flats
import datetime
import os
import cianparser
import pandas as pd

# Импорт для prepare_data
from sklearn.model_selection import train_test_split

# Импорт для Linear_model_train
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Импорт для Save_model
import pickle


def Parse_flats(n_rooms, output_dir='data/raw'):

    df_rooms = pd.DataFrame()

    # Скачивание данных о квартирах с cian
    moscow_parser = cianparser.CianParser(location="Москва")

    for room in n_rooms:
        data = moscow_parser.get_flats(
            deal_type="sale",
            rooms=(room,),  # Кортеж с одним значением
            with_saving_csv=False,
            additional_settings={
                "start_page": 1,
                "end_page": 15,
                "object_type": "secondary"
            })
        
        df = pd.DataFrame(data)
        df_rooms = pd.concat([df_rooms, df], ignore_index=True)

    # Сохраняем CSV
    os.makedirs(output_dir, exist_ok=True)
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_path = f'{output_dir}/{n_rooms[-1]}к_{t}.csv'
    df_rooms.to_csv(csv_path, encoding='utf-8', index=False)

    print("Сохранено")


def Prepare_data(input_dir='data/raw'):

    # Получаю самый последний csv файл
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(input_dir, x)))

    df = pd.read_csv(input_dir+"/"+latest_file)
    df = df[["total_meters","rooms_count","floors_count","floor","price"]]
    df = df.dropna()

    df = df[(df['price'] > 1000) & (df['price'] < 40000000)]
    df = df[(df['total_meters'] > 10) & (df['total_meters'] < 500)]


    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['price']), df['price'], test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def Linear_model_train(X_train, X_test, y_train, y_test):

    modelLR = LinearRegression()
    modelLR.fit(X_train, y_train)
    predictions = modelLR.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    print("R^2:", r2)
    print("MSE:", mse)

    return modelLR
    

def Save_model(model, output_dir='models', name = "Linear"):
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    with open(output_dir+"/"+ name +'Model_'+t+'.pkl', 'wb') as file:
        pickle.dump(model, file)


def Check_model(test_data, input_dir='models'):

    # Получаю самый последний pkl файл
    files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(input_dir, x)))

    with open(input_dir+"/"+latest_file, 'rb') as file:
        loaded_model = pickle.load(file)
        
        column_names = ["total_meters", "rooms_count", "floors_count", "floor"]
        test_data = pd.DataFrame([test_data], columns=column_names)

        predict = int(loaded_model.predict(test_data).round(0))

        print(predict)


def main():

    # 1. Парсинг данных

    # Указываем количество комнат
    number_of_rooms = 4
    # Парсим квартиры с указанным количеством комнат
    variants_of_rooms = [i for i in range(1, number_of_rooms+1)]
    try:
        Parse_flats(variants_of_rooms)
    except Exception as e:
        print(f"Ошибка при парсинге квартир: {e}")


    # 2. Предобработка данных
    X_train, X_test, y_train, y_test = Prepare_data()


    # 3. Обучение модели
    modelLR = Linear_model_train(X_train, X_test, y_train, y_test)

    # 4. Сохранение модели
    Save_model(modelLR)

    # 5. Проверка модели
    Check_model([50, 3, 17, 4])




    

if __name__ == '__main__':
    main()