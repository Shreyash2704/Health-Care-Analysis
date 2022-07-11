from flask import Flask,render_template, url_for ,flash , redirect
import pickle
from flask import request
import numpy as np
import os
from flask import send_from_directory
import joblib
app=Flask(__name__)
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from base64 import b64encode, b64decode
from io import BytesIO

from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# RELATED TO THE SQL DATABASE
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'


with open('heart_disease_model.pkl', 'rb') as file:
      
    # Call load method to deserialze
    myvar = pickle.load(file)
    print(myvar)

new_model = tf.keras.models.load_model('Covid19_Detection.h5')
# Show the model architecture
print(new_model.summary())
@app.route("/")
def home():
    return render_template("index.html")
 

@app.route("/cancer")
def cancer():
    return render_template("cancer.html")


@app.route("/diabetes")
def diabetes():
    #if form.validate_on_submit():
    return render_template("diabetes.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")



@app.route("/kidney")
def kidney():
    #if form.validate_on_submit():
    return render_template("kidney.html")
    
    

@app.route("/covid",methods=['POST','GET'])
def covid():
    if request.method == "POST":
            file = request.files["covid_image"]
   
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                print('upload_image filename: ' + filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                IMG_Load = image.load_img(path,target_size=(256,256))
                N_IMG_Array = image.img_to_array(IMG_Load)
                N_IMG_Array = N_IMG_Array.reshape(1,256,256,3)
                New_Predict = new_model.predict(N_IMG_Array)
                print(New_Predict.argmax(axis=-1))
                result = int(New_Predict.argmax(axis=-1))
                print(result)
                if result == 0:
                    result = "Bacterial Pneumonia detected"
                elif result == 1:
                    result = "Covid 19 detected"
                elif result == 2:
                    result = "Reports are Normal."
                elif result == 3:
                    result = "Viral Pneumonia detected"
                else:
                    result = "Error"
                
        
                return render_template("covidResult.html", prediction=result,desc="Covid 19 Xray")
    else:
        return render_template("covid19.html")

@app.route("/req",methods = ["POST"])
def req():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        print(str(to_predict_list))
        to_predict_list=list(to_predict_list.values())
        print(str(to_predict_list))
        to_predict_list = list(map(float, to_predict_list))
        print(str(to_predict_list))
        return redirect("/")



def ValuePredictor(to_predict_list, size):
    #to_predict = to_predict_list
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==8):#Diabetes 
        d_model = open('diabetes_detection.pkl', 'rb')
        loaded_model = pickle.load(d_model)
        result = loaded_model.predict(to_predict)
    elif(size==30):#Cancer
        loaded_model = joblib.load("model")
        result = loaded_model.predict(to_predict)
    elif(size==12):#Kidney
        loaded_model = joblib.load("model3")
        result = loaded_model.predict(to_predict)
    elif(size==11):#Heart
        loaded_model = joblib.load("model2")
        result =loaded_model.predict(to_predict)
    return [result[0],size]

@app.route('/result',methods = ["POST"])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if(len(to_predict_list)==30):#Cancer
            result = ValuePredictor(to_predict_list,30)
        elif(len(to_predict_list)==8):#Daiabtes
            result = ValuePredictor(to_predict_list,8)
        elif(len(to_predict_list)==12):
            result = ValuePredictor(to_predict_list,12)
        elif(len(to_predict_list)==11):
            result = ValuePredictor(to_predict_list,11)
            
        elif(len(to_predict_list)==10):
            result = ValuePredictor(to_predict_list,10)

    if(int(result[0])==1):
        prediction='Sorry! you are Suffering'
        if result[1] == 8:
            desc = "Diabetes"
        if result[1] == 11:
            desc = "Heart"
        if result[1] == 30:
            desc = "Cancer" 
    else:
        prediction='Congrats ! you are Healthy'
        if result[1] == 8:
            desc = "Diabetes"
        if result[1] == 11:
            desc = "Heart"
        if result[1] == 30:
            desc = "Cancer"

    
    return(render_template("result.html", prediction=prediction,desc=desc))

if __name__ == "__main__":
   app.run(debug = True)