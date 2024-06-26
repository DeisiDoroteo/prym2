# pylint:disable=E0401
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('model_nl.pkl')
app.logger.debug('Modelo de regresión cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        num_reviews = float(request.form['num_reviews'])
        prod_desc = float(request.form['prod_desc'])
        theme_name = float(request.form['theme_name'])
        play_star_rating = float(request.form['play_star_rating'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[num_reviews, prod_desc, theme_name, play_star_rating]],
                               columns=['num_reviews', 'prod_desc', 'theme_name', 'play_star_rating'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar la predicción
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción de precio: {prediction[0]}')
        
        # Devolver la predicción como respuesta JSON
        return jsonify({'precio_predicho': float(prediction[0])})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
