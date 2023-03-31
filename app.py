from flask import Flask, render_template,request,url_for
import pickle
import pandas as pd
import numpy as np

df=pd.read_csv('Cleaned car dset.csv')
app = Flask(__name__)
ML_model=pickle.load(open('Price_predictor2.0.pkl','rb'))
years=list(df['year'].unique())
years.sort(reverse=True)
fuel_type=list(df['fuel_type'].unique())
brands=list(df['company'].unique())
brands.sort()
modelsByBrand = {}
for i in brands:
    key=i
    value=list(df['name'][(df['company']==i)].unique())
    value.sort()
    modelsByBrand[key]=value

@app.route('/')
def index():
    return render_template("index.html", brands=brands, modelsByBrand=modelsByBrand,years=years)

@app.route('/predict',methods=['GET','POST'])
def predict():
    brand=request.form.get('brand')
    model=request.form.get('model')
    year=request.form.get('year')
    Fuel_type=request.form.get('Fuel_type')
    kms_driven=request.form.get('kms_driven')
    prediction=ML_model.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],
                              data=np.array([model,brand,year,kms_driven,Fuel_type]).reshape(1,5)))
    output=int(prediction)
    return render_template('results.html',predicted=str(output),brand=brand,model=model,year=year,Fuel_type=Fuel_type, kms_driven=kms_driven)


if __name__ == "__main__":
    app.run()