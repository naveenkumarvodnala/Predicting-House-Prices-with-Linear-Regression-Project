from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('housing_data.pkl', 'rb') as file:
    lr = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form submission
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    sqft_footage= int(request.form['sqft_footage'])
    price = int(request.form['price'])
    

    # Prepare the data for prediction
    features = np.array([[bedrooms, bathrooms, sqft_footage, price]])
    
    # Make a prediction
    predicted_price = lr.predict(features)[0]
    

    return render_template('index.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
