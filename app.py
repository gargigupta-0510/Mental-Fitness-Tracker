from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('mental_fitness_model.pkl')
country_encoder = joblib.load('country_encoder.pkl')  # Load the saved encoder
scaler = joblib.load(r'C:\Users\lenovo\OneDrive\Documents\Gargi\python\MentalFitnessTracker\scaler.pkl')  # load saved scaler
feature_columns = joblib.load(r'C:\Users\lenovo\OneDrive\Documents\Gargi\python\MentalFitnessTracker\feature_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Decode country name
        country_name = request.form['country']
        country_encoded = country_encoder.transform([country_name])[0]
        
        inputs = [
            country_encoded,
            int(request.form['year']),
            (float(request.form['schizo'])),
            (float(request.form['bipolar'])),
            (float(request.form['anxiety'])),
            (float(request.form['eating'])),
            (float(request.form['drug'])),
            (float(request.form['depression'])),
            (float(request.form['alcohol']))
        ]
        print(f"Raw inputs: {inputs}")


        scaled_input = scaler.transform([inputs])
        print(f"Scaled inputs: {scaled_input[0]}")
        prediction = model.predict(scaled_input)[0]
        #prediction = (model.predict([inputs])[0])*10
        prediction_rounded = round(prediction, 2)
        print(f"Predicted mental fitness score: {prediction_rounded}")


        # Interpretation
        if prediction_rounded <= 2.0:
            message = "🚨 Poor Mental Health: Immediate support recommended."
        elif prediction_rounded <= 4.0:
            message = "⚠️ Moderate Mental Health: Monitor and practice self-care."
        elif prediction_rounded < 6.0:
            message = "🔄 Moderate Mental Health: Focus on self-care and monitoring your wellbeing."
        elif prediction_rounded < 8.0:
            message = "✅ Good Mental Health: Maintain your healthy lifestyle and habits."
        else:
            message = "🌟 Excellent Mental Health: You're doing great!"

        return render_template('result.html', prediction=prediction_rounded, message=message)

    except ValueError:
        return "Invalid country name. Please enter a valid country from the dataset."
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
