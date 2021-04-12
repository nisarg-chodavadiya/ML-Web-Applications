import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('concrete.pkl', 'rb'))


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    cement = request.form['cement']
    blast_furnace_slag = request.form['blast_furnace_slag']
    fly_ash = request.form['fly_ash']
    water = request.form['water']
    superplasticizer = request.form['superplasticizer']
    coarse_aggregate = request.form['coarse_aggregate']
    fine_aggregate = request.form['fine_aggregate']
    age = request.form['age']

    features = np.array([[cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age]])
    prediction = model.predict(features)

    return render_template('after.html', prediction_text='Predicted Concrete Compressive Strength = {} \u00B1 4.0 MPa'.format(prediction[0]))


if __name__ == '__main__':
    app.run(debug=True)
