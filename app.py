from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model1.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index1.html")

@app.route('/predict',methods=['POST'])
def predict():
    

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    if output==0:
        return render_template('index1.html', prediction_text='SPECIES OF IRIS IS SETOSA')
    elif output==1:
        return render_template('index1.html', prediction_text='SPECIES OF IRIS IS VERSICOLOR')
    else:
        return render_template('index1.html', prediction_text='SPECIES OF IRIS IS VIRGINICA')


if __name__ == '__main__':
    app.run(debug=True)
