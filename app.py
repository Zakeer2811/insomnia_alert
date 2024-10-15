from flask import Flask, request, jsonify, render_template, flash
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the models
model_path = os.path.join(os.path.dirname(__file__), 'model', 'epics_model.joblib')
kmeans_model_path = os.path.join(os.path.dirname(__file__), 'model', 'kmeans_model.joblib')
scaler_path = os.path.join(os.path.dirname(__file__), 'model', 'scaler.joblib')

# Load RandomForest and KMeans models
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    model = None
    print(f"Model file not found at {model_path}. Please check the path.")
except Exception as e:
    model = None
    print(f"An error occurred while loading the RandomForest model: {e}")

try:
    kmeans_model = joblib.load(kmeans_model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    kmeans_model = None
    scaler = None
    print(f"Model file not found at {kmeans_model_path}. Please check the path.")
except Exception as e:
    kmeans_model = None
    scaler = None
    print(f"An error occurred while loading the KMeans model: {e}")

# Define required columns for prediction
required_columns = [
    'Age', 'Sleep duration', 'Sleep efficiency',
    'REM sleep percentage', 'Deep sleep percentage',
    'Light sleep percentage', 'Awakenings',
    'Caffeine consumption', 'Alcohol consumption',
    'Exercise frequency'
]

@app.route('/')
def home():
    return render_template('index.html')

# Define a route for predictions (RandomForest)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        features = [
            data['Age'], data['Sleep duration'], data['REM sleep percentage'],
            data['Deep sleep percentage'], data['Light sleep percentage'],
            data['Awakenings'], data['Caffeine consumption'],
            data['Alcohol consumption'], data['Exercise frequency']
        ]
        prediction = model.predict([features])
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

# Handle CSV file upload and predict insomnia (KMeans)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.csv'):
                if scaler is None:
                    flash("Scaler model not loaded. Please check the model path.")
                    return render_template('index.html')
                try:
                    df = pd.read_csv(file)
                    # Ensure only required columns are used
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        flash(f"Missing columns in the uploaded CSV: {', '.join(missing_columns)}")
                        return render_template('index.html')

                    # Impute missing values
                    imputer = SimpleImputer(strategy='mean')
                    df[required_columns] = imputer.fit_transform(df[required_columns])

                    input_array = df[required_columns].values
                    input_scaled = scaler.transform(input_array)  # Scale the input data
                    predictions = kmeans_model.predict(input_scaled)

                    # Prepare the results to display in the template
                    results = {i: "You have insomnia." if pred == 1 else "You do not have insomnia." for i, pred in enumerate(predictions)}
                    return render_template('index.html', results=results)
                except Exception as e:
                    flash(f"Error processing CSV file: {e}")
            else:
                flash("Please upload a valid CSV file.")
        else:
            input_data = {}
            if scaler is None:
                flash("Scaler model not loaded. Please check the model path.")
                return render_template('index.html')
            try:
                for column in required_columns:
                    value = request.form[column]
                    input_data[column] = float(value)

                input_array = np.array([[input_data[col] for col in required_columns]])
                input_scaled = scaler.transform(input_array)  # Scale the input data
                prediction = kmeans_model.predict(input_scaled)
                result = "You have insomnia." if prediction[0] == 1 else "You do not have insomnia."
                return render_template('index.html', result=result)
            except Exception as e:
                flash(f"Error during prediction: {e}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
