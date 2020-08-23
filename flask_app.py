# import libraries
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import request,Flask,jsonify,render_template
import pickle

# initialize flask application
app = Flask(__name__)

#load knn_model
model = pickle.load(open('knn_model.pickle','rb'))

# define method for different urls
@app.route('/') #give url for your homepage
def home():     # call this method when we go to (‘localhost:5000/’) url
	return render_template('index.html') #return index.html file

@app.route('/predict',methods=['POST']) #create predict url
def predict(): # call this method when we go to (‘/predict’) url
	input = [float(x) for x in request.form.values()]  #get value from the form which is present in index.html
	print(input)
	input_array = (np.array(input)).reshape(1,-1) #convert values into array
	print(input_array)
	# scale = StandardScaler()
	# input_array_normalize = scale.fit_transform(input_array) 
	prediction = model.predict(input_array) #predict value based on the input array from our knn model
	print(prediction)
	return render_template('index.html',prediction='{}'.format(prediction)) #return index.html file with prediction value

if __name__ == "__main__":
    app.run(debug=True)
