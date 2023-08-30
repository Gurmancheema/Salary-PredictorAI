from flask import Flask,request,jsonify,app,render_template,url_for
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open('randomforestregressor.pkl','rb'))
encoder = pickle.load(open('onehotlabelencoder.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Predict', methods  = ['POST'])

def predict():
    data =  [25,'Male','PhD','Junior Data Scientist',2,'Canada','Asian']
    sample_test_data = pd.DataFrame([data],columns =['Age','Gender','Education Level','Job Title','Years of Experience','Country','Race'])
    sample_test_encoded = encoder.transform(sample_test_data[['Gender','Job Title','Education Level','Country','Race']])
    sample_test_encoded_df = pd.DataFrame(sample_test_encoded.toarray(),columns=encoder.get_feature_names_out(['Gender','Job Title','Education Level','Country','Race']))
    sample_selected_columns_df = sample_test_data[['Age','Years of Experience']]
    sample_test_df = pd.concat([sample_selected_columns_df,sample_test_encoded_df], axis=1)

    #prediction part
    sample_prediction = model.predict(sample_test_df)
    #print("The predicted salary for this individual is : ${:,.2f}".format(round(sample_prediction.item(),2)))
    return render_template('home.html',prediction_text = "The predicted salary for this individual is : ${:,.2f}".format(round(sample_prediction.item(),2)))

@app.route('/predict_api', methods = ['POST'])

def predict_api():

    data = request.form.values()
    sample_test_data = pd.DataFrame([data],columns =['Age','Gender','Education Level','Job Title','Years of Experience','Country','Race'])
    sample_test_encoded = encoder.transform(sample_test_data[['Gender','Job Title','Education Level','Country','Race']])
    sample_test_encoded_df = pd.DataFrame(sample_test_encoded.toarray(),columns=encoder.get_feature_names_out(['Gender','Job Title','Education Level','Country','Race']))
    sample_selected_columns_df = sample_test_data[['Age','Years of Experience']]
    sample_test_df = pd.concat([sample_selected_columns_df,sample_test_encoded_df], axis=1)

    #prediction part
    sample_prediction = model.predict(sample_test_df)
    #print("The predicted salary for this individual is : ${:,.2f}".format(round(sample_prediction.item(),2)))
    return render_template('home.html',prediction_text = "The predicted salary for this individual is : ${:,.2f}".format(round(sample_prediction.item(),2)))





if(__name__) =='__main__':
    app.run(debug=True)