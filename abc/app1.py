from flask import Flask, render_template, request
import pickle
import xgboost as xgb
import numpy as np  # Import NumPy for array operations

app = Flask(__name__)

# Load the model
model = pickle.load(open("xgbc.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user input
        gender = request.form['gender']
        age = request.form['age']
        occupation = request.form['occupation']
        sleep = request.form['sleep']
        qualitysleep = request.form['qualitysleep']
        activity = request.form['activity']
        stress = request.form['stress']
        heartRate = request.form['heartRate']
        dailySteps = request.form['dailySteps']
        bmi = request.form['bmi']
        Systolic = request.form['Systolic']
        Diastolic = request.form['Diastolic']

       # Perform one-hot encoding for categorical variables
        #gender_male = 1 if gender == "Male" else 0
        #gender_female = 1 if gender == "Female" else 0
        # Repeat the encoding for other categorical variables

        # Create a list of features for prediction
        final_features = [
            gender, 
            age, sleep, qualitysleep, activity, stress, heartRate, dailySteps, bmi, Systolic, Diastolic
        ]
        

        # Make the prediction using your model
        prediction = model.predict([final_features])[0]

        # Map prediction values to corresponding sleep disorders
        #sleep_disorder_map = {0: "insomnia", 1: "none", 2: "sleep apnea"}
        #prediction_text = sleep_disorder_map.get(prediction, "Unknown")

        # Render the 'prediction.html' template and pass the prediction_text as a variable
        return render_template('prediction.html',result=prediction)

    #return render_template('prediction.html')  # Render the input form initially

if __name__ == '__main__':
    app.run(debug=True)
