from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and column names
with open(r'model.pkl', 'rb') as file:
    model = pickle.load(file)

with open(r'columns.pkl', 'rb') as file:
    columns = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input from the user
    gender = request.form.get('gender')
    weight = float(request.form.get('weight'))
    age = float(request.form.get('age'))
    smoking = float(request.form.get('smoking'))
    yellow_fingers = float(request.form.get('yellow_fingers'))
    anxiety = float(request.form.get('anxiety'))
    family_history = float(request.form.get('family_history'))
    chronic_disease = float(request.form.get('chronic_disease'))
    fatigue = float(request.form.get('fatigue'))
    allergy = float(request.form.get('allergy'))
    wheezing = float(request.form.get('wheezing'))
    alcohol_consuming = float(request.form.get('alcohol_consuming'))
    coughing = float(request.form.get('coughing'))
    shortness_of_breath = float(request.form.get('shortness_of_breath'))
    swallowing_difficulty = float(request.form.get('swallowing_difficulty'))
    chest_pain = float(request.form.get('chest_pain'))

    #Gender
    female = 0
    male = 0
    if int(gender) == 1:
        male = float(1)
    else:
        female = float(1)

    # Convert the input to a numpy array
    input_array = np.array([age,weight, smoking, yellow_fingers, anxiety, family_history,
                            chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                            coughing, shortness_of_breath, swallowing_difficulty, chest_pain, female, male]).reshape(1, 17)

    # Create a dataframe with the input features and column names
    input_df = pd.DataFrame(input_array, columns=columns)

    # Use the trained model to make predictions
    prediction = model.predict(input_df)
    prediction = prediction * 100

    return render_template('index.html', prediction_text='The probability of having lung cancer is {}%.'.format(prediction), input_array=input_array, columns=columns, percent=int(prediction))

#if __name__ == '__main__':
 #   app.run(debug=True)