from flask import Flask, request, jsonify, render_template
import logging
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Настройка логирования с UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./pabd25/logs/app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_latest_model(directory):
    """Находит последнюю обученную модель"""
    try:
        files = [f for f in os.listdir(directory) if f.endswith(('.pkl', '.joblib'))]
        if not files:
            return None
        latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(directory, f)))
        return os.path.join(directory, latest_file)
    except Exception as e:
        logger.error(f"Error finding model: {e}")
        return None

# Инициализация модели глобально
model = None

def load_model():
    """Загружает модель при старте приложения"""
    global model
    model_path = get_latest_model('./pabd25/models')
    if model_path:
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            if hasattr(model, 'feature_names_in_'):
                logger.info(f"Model expects features: {list(model.feature_names_in_)}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            model = None
    else:
        logger.error("No model found in directory")

# Загружаем модель при старте
load_model()

def format_rubles(amount: float) -> str:
    """Форматирует сумму в рублях"""
    amount = round(amount)
    if amount >= 1_000_000:
        return f"{amount // 1_000_000} млн {amount % 1_000_000 // 1_000} тыс руб"
    elif amount >= 1_000:
        return f"{amount // 1_000} тыс руб"
    return f"{amount} руб"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/numbers', methods=['POST'])
@app.route('/api/numbers', methods=['POST'])
def process_numbers():
    logger.info("Request received for /api/numbers")
    data = request.get_json()

    if not data:
        logger.error("Error: no data received")
        return {'status': 'error', 'message': 'Ошибка: данные не получены'}, 400
        
    try:
        num1 = float(data.get('number1', 0))
        num2 = int(data.get('number2', 1))
        num3 = int(data.get('number3', 1))
        num4 = int(data.get('number4', 1))
        
        logger.info(f"Input data: {num1}m², {num2} rooms, floor {num4}/{num3}")

        if model is None:
            raise RuntimeError("Модель не загружена")

        # Создаем DataFrame с правильными признаками
        input_data = {
            'total_meters': [num1],
            'floor': [num4],
            'floors_count': [num3],
            'rooms_count': [num2]
        }
        
        if hasattr(model, 'feature_names_in_') and 'relative_floor' in model.feature_names_in_:
            input_data['relative_floor'] = [num4 / max(1, num3)]
        
        input_df = pd.DataFrame(input_data)
        
        if hasattr(model, 'feature_names_in_'):
            input_df = input_df[list(model.feature_names_in_)]
        
        prediction = model.predict(input_df)[0]
        predicted_price = (prediction * 1000000).round(0)
        formatted_price = format_rubles(predicted_price)
        
        return {
            'status': 'success',
            'price': int(predicted_price),
            'formatted_price': formatted_price,
            'message': f'Предсказанная цена: {formatted_price}'
        }

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'message': f'Ошибка: {str(e)}'
        }, 400

if __name__ == '__main__':
    logger.info("Starting server...")
    app.run(host='0.0.0.0', port=5000, debug=True)