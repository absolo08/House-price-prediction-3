from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('grid.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = ''
    if request.method == 'POST':
        try:
            # Collect numeric inputs
            bedrooms = float(request.form['bedrooms'])
            bathrooms = float(request.form['bathrooms'])
            sqft_living = float(request.form['sqft_living'])
            sqft_lot = float(request.form['sqft_lot'])
            sqft_above = float(request.form['sqft_above'])
            sqft_basement = float(request.form['sqft_basement'])
            yr_built = float(request.form['yr_built'])
            yr_renovated = float(request.form['yr_renovated'])

            # Collect dropdown selections (converted to float)
            floors = float(request.form['floors'])
            condition = float(request.form['condition'])
            city = float(request.form['city'])
            statezip = float(request.form['statezip'])

            # Combine all input features into the correct order expected by the model
            features = np.array([[
                bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                condition, sqft_above, sqft_basement,
                yr_built, yr_renovated, city, statezip
            ]])

            # Predict using the trained model
            predicted_price = model.predict(features)[0]
            price = round(predicted_price, 2)

            # Return formatted result to template
            prediction = f"🏠 Estimated House Price: ${price:,.2f}"

        except ValueError:
            prediction = '⚠️ Please enter valid numeric values.'
        except Exception as e:
            prediction = f'❌ Error: {str(e)}'

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
