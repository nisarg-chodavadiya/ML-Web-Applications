import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('cell.pkl', 'rb'))


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    clump = request.form['clump']
    unifsize = request.form['unifsize']
    unifshape = request.form['unifshape']
    margadh = request.form['margadh']
    singepisize = request.form['singepisize']
    barenuc = request.form['barenuc']
    blandchrom = request.form['blandchrom']
    normnucl = request.form['normnucl']
    mit = request.form['mit']

    features = np.array([[clump, unifsize, unifshape, margadh, singepisize, barenuc, blandchrom, normnucl, mit]])
    prediction = model.predict(features)

    return render_template('after.html', prediction_text='Cell Class is "{}".'.format(prediction))


if __name__ == '__main__':
    app.run(debug=True)
