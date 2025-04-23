from flask import Flask, request, jsonify, render_template
import logging

logger = logging.getLogger('my_logger')

file_handler = logging.FileHandler('app.log', mode='w+')
logger.addHandler(file_handler)


logger.warning("Program start")

app = Flask(__name__)

# Маршрут для отображения формы
@app.route('/')
def index():
    return render_template('index.html')

# Маршрут для обработки данных формы
@app.route('/api/numbers', methods=['POST'])
def process_numbers():
    # Здесь можно добавить обработку полученных чисел
    # Для примера просто возвращаем их обратно

    logger.warning("")

    data = request.get_json()

    if not data:
        return {'status': 'error', 'message': 'Ошибка при получение данных. Данные не пришли'}
        
    num1 = data.get('number1')
    num2 = data.get('number2')
    num3 = data.get('number3')
    num4 = data.get('number4')
    logger.warning('data recieved')
    logger.warning('Square: '+num1)
    logger.warning('Num of rooms: '+num2)
    logger.warning('Num of floors: '+num3)
    logger.warning('Floor: '+num4)

    try:
        num1 = float(num1)
        num2 = int(num2)
        num3 = int(num3)
        num4 = int(num4)
    except (ValueError, TypeError):
        logger.warning("Не получилось преобразовать введенные данные")
        return {'status': 'error', 'message': 'Ошибка при обработке данных'}
        
    
    print("\n=== Получены данные ===")
    print(f"Площадь квартиры: {data.get('number1')} м²")
    print(f"Количество комнат: {data.get('number2')}")
    print(f"Этажей в доме: {data.get('number3')}")
    print(f"Этаж квартиры: {data.get('number4')}")
    print("=====================\n")
    

    logger.warning("Program ended")
    return {'status': 'success', 'data': 'Числа успешно обработаны'}

if __name__ == '__main__':
    app.run(debug=True)
    
