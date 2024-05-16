import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('customer.h5')  # Load the trained model
# Define mapping dictionaries for categorical variables
# internet_service_mapping = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
payment_method_mapping = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
 # Retrieve form data
    input_data = {
        "Gender": request.form['Gender'],
        "SeniorCitizen": int(request.form['SeniorCitizen']),
     #   "Partner": request.form['Partner'],
        "Dependents": request.form['Dependents'],
        "Tenure": int(request.form['Tenure']),
   #     "Multiple Lines": request.form['Multiple Lines'],
  #      "Phone Service": request.form['Phone Service'],
   #     "Internet Service": request.form['Internet Service'],
    #    "Online Security": request.form['Online Security'],

   #     "Online Backup": request.form['Online Backup'],
    #    "Device Protection": request.form['Device Protection'],

   #     "Tech Support": request.form['Tech Support'],
  #      "Streaming TV": request.form['Streaming TV'],

   #     "Streaming Movies": request.form['Streaming Movies'],
        "Contract": request.form['Contract'],

    #    "Paperless Billing": request.form['Paperless Billing'],
        "Payment Method": request.form['Payment Method'],

       # "Monthly Charges": float(request.form['Monthly Charges']),
       # "Total Charges":float(request.form['Total Charges'])

 }

    # Dummy encoding for categorical variables (replace with actual encoding)
    input_data["Gender"] = 1 if input_data["Gender"].lower() == "male" else 0
  # Dummy encoding for binary variables
    binary_variables = [ "Dependents"]

    for var in binary_variables:
        input_data[var] = 1 if input_data[var].lower() == "yes" else 0
         # Perform one-hot encoding for categorical variables
    #input_data["Internet Service"] = internet_service_mapping.get(input_data["Internet Service"], 0)
    input_data["Contract"] = contract_mapping.get(input_data["Contract"], 0)
    input_data["Payment Method"] = payment_method_mapping.get(input_data["Payment Method"], 0)
    input_df = pd.DataFrame([input_data])

    cols_to_scale = ['Tenure','Contract','Payment Method']
    scaler = MinMaxScaler()
    input_df[cols_to_scale] = scaler.fit_transform(input_df[cols_to_scale])
    # Convert input_data into a DataFrame or 2D array
    # Call function to predict customer churn
    prediction = model.predict(input_df)
    prediction_status = "Staying" if prediction[0][0] >= 0.5 else "Leaving"

    return render_template('home.html', prediction_text="Customer Staying Status: {}".format(prediction_status))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    int_features = [float(x) for x in data.values()]
    final_features = np.array(int_features).reshape(1, -1)  # Reshape features for model input
    prediction = model.predict(final_features)
    output = "Staying" if prediction[0][0] >= 0.5 else "Leaving"
    return jsonify({"prediction": output})

if __name__ == '__main__':
    app.run(debug=True)
