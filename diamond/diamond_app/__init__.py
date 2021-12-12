from flask import Flask, render_template, request
import numpy as np

import pickle

model = None
with open('model.pkl','rb') as pickle_file:
    model = pickle.load(pickle_file)

def create_app():
    app = Flask(__name__)

    @app.route('/')
    def home():
        return render_template('home.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        data0 = request.form['shape']
        data1 = request.form['carat']
        data2 = request.form['cut']
        data3 = request.form['color']
        data4 = request.form['clarity']
        data5 = request.form['report']
        data6 = request.form['type']

        y_pred = model.predict(
            [[int(data0), float(data1), int(data2), int(data3), int(data4), int(data5), int(data6)]]
        )
        return render_template('result.html', data = int(y_pred))
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", debug=True)
