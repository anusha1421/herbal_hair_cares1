from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import os
from PIL import Image
import matplotlib.pyplot as plt

# Importing model prediction functions (for hair care-related prediction)
from model_predict import herbal_hair_care
from model_predict_hair_loss import herbal_hair_care2

# Load the student placement prediction model
model_path = 'int_features3.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Disease dictionaries
disease_dic = ["Alopecia", "Folliculitis", "Psoriasis"]
disease_dic2 = ["stage0", "stage1", "stage2", "stage3", "stage4"]

# Initializing the Flask app
app = Flask(__name__)

# Home page for general navigation
@app.route('/')
def home():
    return render_template('index.html')

# Disease prediction page for hair care-related diseases
@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Classification'

    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)

        img1 = file.read()
        prediction = herbal_hair_care(img1)
        prediction = str(disease_dic[prediction])

        if prediction == "Alopecia":
            precaution = "Alopecia is related to hair loss. Consult a dermatologist."
            products_recommendation = ["Minoxidil", "Biotin", "Gentle hair products"]
            product_routine = "Use Minoxidil twice a day. Take Biotin daily."
        elif prediction == "Folliculitis":
            precaution = "Maintain good hygiene."
            products_recommendation = ["Antibacterial soap", "Topical antibiotics", "Tea tree oil"]
            product_routine = "Use products twice daily."
        elif prediction == "Psoriasis":
            precaution = "Consult a dermatologist."
            products_recommendation = ["Moisturizers", "Corticosteroids", "Salicylic acid products"]
            product_routine = "Follow prescribed routines."

        return render_template('disease-result.html', prediction=prediction, precaution=precaution, 
                               products_recommendation=products_recommendation, product_routine=product_routine, title=title)

    return render_template('disease.html', title=title)

# Hair loss classification page
@app.route('/disease-predict2', methods=['GET', 'POST'])
def disease_prediction2():
    title = 'Hair Loss Classification'

    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template('disease2.html', title=title)

        img1 = file.read()
        prediction = herbal_hair_care2(img1)
        prediction = str(disease_dic2[prediction])

        precautions_and_recommendations = {
            "stage0": ("Use prescribed products.", ["Minoxidil", "Biotin", "Gentle hair care"]),
            "stage1": ("Use prescribed products.", ["Antibacterial soap", "Antibiotics", "Tea tree oil"]),
            "stage2": ("Use prescribed products.", ["Moisturizers", "Corticosteroids", "Salicylic acid"]),
            "stage3": ("Continue prescribed treatments.", ["Moisturizers", "Corticosteroids", "Salicylic acid"]),
            "stage4": ("Consider advanced options like wigs.", ["Wigs", "Scalp treatments"])
        }

        precaution, products_recommendation = precautions_and_recommendations.get(prediction, ("", []))

        # Log user prediction into CSV
        file_name = "user_predictions.csv"
        if os.path.isfile(file_name):
            df = pd.read_csv(file_name)
        else:
            df = pd.DataFrame(columns=['stage', 'treatment', 'date'])

        current_datetime = pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S")
        new_data = {'stage': [prediction], 'treatment': [products_recommendation], 'date': [current_datetime]}
        
        df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)
        df.to_csv(file_name, index=False)

        return render_template('disease-result2.html', prediction=prediction, precaution=precaution, 
                               products_recommendation=products_recommendation, title=title)

    return render_template('disease2.html', title=title)

# Student placement prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form for student placement
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    output = 'Placed' if prediction[0] == 1 else 'Not Placed'

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

# Plotting function for hair disease prediction
@app.route('/plot')
def plot_graph():
    file_name = "user_predictions.csv"
    
    if not os.path.isfile(file_name):
        return render_template('plot.html', image_file='/static/images/no_data.png')

    df = pd.read_csv(file_name)
    if df.empty:
        return render_template('plot.html', image_file='/static/images/no_data.png')

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    plt.figure(figsize=(10, 6))
    stage_colors = {'stage0': 'blue', 'stage1': 'green', 'stage2': 'orange', 'stage3': 'red', 'stage4': 'purple'}

    for stage, color in stage_colors.items():
        stage_df = df[df['stage'] == stage]
        plt.plot_date(stage_df.index, [stage]*len(stage_df), linestyle='-', marker='o', label=stage, color=color)

    plt.legend()
    plt.xlabel('Dates')
    plt.ylabel('Stages')
    plt.title('Stages Distribution Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/images/bar_plot.png')
    plt.close()

    return render_template('plot.html', image_file='/static/images/bar_plot.png')

@app.route('/def_suggestion')
def def_suggestion():
    # Your logic here
    return render_template('herbalsolution.html')


if __name__ == "__main__":
    app.run(debug=True)
