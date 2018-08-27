import flask
from flask import Flask, render_template, request
from test import make_prediction 
from keras.models import load_model

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
   return flask.render_template('index.html')

@app.route("/predict", methods=['POST'])
def prediction():
	if request.method=='POST':
		text = request.form['text']
	
		if not text:
			return flask.render_template('index.html', text="No text for prediction")	
		if len(text) < 100:
			return flask.render_template('index.html', text="Please input at least 100 character text")	

		new_text = make_prediction(text)
		return flask.render_template('index.html', text=new_text)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8000, debug=True)