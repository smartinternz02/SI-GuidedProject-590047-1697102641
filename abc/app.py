from flask import Flask, render_template, request
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
model = pickle.load(open('xgbc.pkl', 'rb'))


# Define the route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route to handle the form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get the values entered in the form
    gender = float(request.form['Gender'])
    age = float(request.form['Age'])
    occupation = float(request.form['Occupation'])
    sleep_duration = float(request.form['Sleep Duration'])
    quality_of_Sleep = float(request.form['Quality of Sleep'])
    physical_activity_Level= float(request.form['Physical Activity Level'])
    stress_Level= float(request.form['Stress Level'])
    bmi_Category= float(request.form['BMI Category'])
    heart_Rate= float(request.form['Heart Rate'])
    Daily_Steps= float(request.form['Daily Steps'])
    Systolic= float(request.form['Systolic'])
    Diastolic= float(request.form['Diastolic'])
    # Perform the addition
    result = model.predict([[gender,age,occupation,sleep_duration,quality_of_Sleep,physical_activity_Level,stress_Level,bmi_Category,heart_Rate,Daily_Steps,Systolic,Diastolic]])
    print(result)


    # Render the result in a new template.
    return render_template('result.html', result=result)

if __name__ == '__main__':
     app.run(debug=True)



