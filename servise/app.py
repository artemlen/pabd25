from flask import Flask, request, jsonify, render_template
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('servise/app.log'), 
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка модели при старте приложения
model_path = "C:\\Users\\user\\Институт\\PABD25\\Проект_2\\models\\model_house_prise.pkl"
model = joblib.load(model_path)

# Маршрут для отображения формы
@app.route('/')
def index():
    return render_template('index.html')

# Маршрут для обработки данных формы
@app.route('/api/numbers', methods=['POST'])
def process_numbers():

    logger.info("Request received for /api/numberss")
    data = request.get_json()

    if not data:
        logger.error("Error: no data received")
        return {'status': 'error', 'message': 'Ошибка при получении данных. Данные не пришли'}
        
    num1 = data.get('number1')
    num2 = data.get('number2')
    num3 = data.get('number3')
    num4 = data.get('number4')

    try:
        num1 = float(num1)
        num2 = int(num2)
        num3 = int(num3)
        num4 = int(num4)
        logger.info(f"The data has been successfully converted: {num1}, {num2}, {num3}, {num4}")
    except (ValueError, TypeError) as e:
        logger.error(f"Error in data processing: {e}")
        return {'status': 'error', 'message': 'Ошибка при обработке данных'}
    
    logger.info("=== Data received ===")
    logger.info(f"The area of the apartment: {num1}")
    logger.info(f"Number of rooms: {num2}")
    logger.info(f"Floors in the house: {num3}")
    logger.info(f"Apartment floor: {num4}")
    logger.info("=====================\n")

    if model is not None:
        try:
            # Преобразуем данные в 2D массив для модели
            input_df = pd.DataFrame([[num1, num4, num3, num2]], columns=['total_meters', 'floors_count', 'floor', 'rooms_count'])
            prediction = model.predict([[num1, num4, num3, num2]])[0]
            predicted_price = round(float(prediction), 2)
            logger.info(f"Predicted price: {predicted_price}")
            return {'status': 'success', 'message': f'Предсказанная цена: {predicted_price} руб.'}
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'status': 'error', 'message': 'Ошибка при предсказании цены'}
    else:
        logger.error("Model not available")
        return {'status': 'error', 'message': 'Модель не загружена'}

    return {'status': 'success', 'message': 'Цена: ' + str(300000 *num1)}

if __name__ == '__main__':
    logger.info("The server is running")
    app.run(debug=True)

