from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
with open('House_Price_Prediction_Model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    try:
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        stories = int(request.form['stories'])
        mainroad = request.form['mainroad']
        guestroom = request.form['guestroom']
        basement = request.form['basement']
        hotwaterheating = request.form['hotwaterheating']
        airconditioning = request.form['airconditioning']
        parking = int(request.form['parking'])
        prefarea = request.form['prefarea']
        furnishingstatus = request.form['furnishingstatus']
        
        # Create a data frame for the model input
        input_data = pd.DataFrame({
            'area': [area],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'stories': [stories],
            'mainroad': [1 if mainroad == 'yes' else 0],
            'guestroom': [1 if guestroom == 'yes' else 0],
            'basement': [1 if basement == 'yes' else 0],
            'hotwaterheating': [1 if hotwaterheating == 'yes' else 0],
            'airconditioning': [1 if airconditioning == 'yes' else 0],
            'parking': [parking],
            'prefarea': [1 if prefarea == 'yes' else 0],
            'furnishingstatus': [1 if furnishingstatus == 'furnished' else 2 if furnishingstatus == 'semi-furnished' else 3]
        })

        # Predict the house price using the model
        prediction = model.predict(input_data)[0]
        
        # Return the prediction to the client-side
        return jsonify({'predicted_price': round(prediction, 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
