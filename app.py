from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.exception import CustomException
from src.logger import logging
import os

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                gender = request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch = request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=request.form.get('reading_score'),
                writing_score=request.form.get('writing_score'),
                
            )
            pred_df = data.get_data_as_data_frame()
            
            print(pred_df)
            
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('home.html',results = results)
        
        except Exception as e:
            logging.error(f"Error occured while predicting the data:{e}")
            return render_template('home.html',results = e)
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)