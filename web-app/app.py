import numpy as np  
from flask import Flask, request, render_template
import pickle

#initialize Flask
app = Flask(__name__)
model = pickle.load(open('ufo-model.pkl','rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # 1. Get the values from the form
    int_features= [int(x) for x in request.form.values()]
    # 2. Turn them into a format the model understands (Numpy array)
    final_features = [np.array(int_features)]
    # 3. Ask the model for a prediction
    prediction = model.predict(final_features)

    output= prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"] 
    country_name = countries[output]

    
    # 4. Show the result on the screen
    return render_template("index.html", prediction_text="Likely country: {}".format(country_name)
    )
if __name__ == "__main__":
    app.run(debug=True)
