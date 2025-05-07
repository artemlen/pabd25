"""Parse data from cian.ru"""
import datetime
import os
import cianparser
import pandas as pd

moscow_parser = cianparser.CianParser(location="Москва")

def parse_flats(n_rooms, output_dir='data/raw'):
    """
    Парсит данные для указанного количества комнат
    """
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_path = f'{output_dir}/{n_rooms}к_{t}.csv'
    
    # Создаем директорию, если ее нет
    os.makedirs(output_dir, exist_ok=True)
    
    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=(n_rooms,),  # Кортеж с одним значением
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 50,
            "object_type": "secondary"
        })
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, encoding='utf-8', index=False)
    print(f"Сохранено {len(df)} записей для {n_rooms}-комнатных квартир в {csv_path}")
    return df

def main():
    """Основная функция для парсинга всех типов квартир"""
    # Парсим квартиры с 1 до 3 комнат
    for n_rooms in [1, 2, 3]:
        try:
            parse_flats(n_rooms)
        except Exception as e:
            print(f"Ошибка при парсинге {n_rooms}-комнатных квартир: {e}")

if __name__ == '__main__':
    main()
