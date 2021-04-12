import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('Iris.pkl', 'rb'))


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    sl = request.form['sl']
    sw = request.form['sw']
    pl = request.form['pl']
    pw = request.form['pw']
    features = np.array([[sl, sw, pl, pw]])
    prediction = model.predict(features)
    return render_template('after.html', prediction_text='Iris Species is "{}".'.format(prediction[0]))


if __name__ == '__main__':
    app.run(debug=True)
