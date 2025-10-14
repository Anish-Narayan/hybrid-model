from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# --- 1. Synthetic Data Generation and In-Memory Model Training ---

# Define the features provided by the user for model training and prediction
FINAL_FEATURE_NAMES = [
    'Application mode', 'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (approved)', 
    'Course', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)', 
    "Mother's occupation", "Father's occupation", 'Age at enrollment', 'Previous qualification', 
    'Curricular units 2nd sem (evaluations)', 'Scholarship holder', "Mother's qualification", 
    'Debtor', 'Curricular units 1st sem (evaluations)', 'Gender', 
    'Curricular units 2nd sem (without evaluations)', 'Admission grade', 
    'Curricular units 2nd sem (enrolled)', 'Tuition fees up to date'
]

# --- Default Values (Based on user request) ---
# These are fixed default values for the 20 selected features.
DEFAULT_VALUES = {
    # Assuming the first numeric sequence from your input corresponds to these features
    'Application mode': 2.0,
    'Curricular units 2nd sem (grade)': 10.8,
    'Curricular units 2nd sem (approved)': 6.0,
    'Course': 9119.0, # A typical course ID value
    'Curricular units 1st sem (approved)': 11.0,
    'Curricular units 1st sem (grade)': 10.0,
    "Mother's occupation": 6.0,
    "Father's occupation": 6.0,
    'Age at enrollment': 19.0,
    'Previous qualification': 1.0,
    'Curricular units 2nd sem (evaluations)': 11.0,
    'Scholarship holder': 1.0,
    'Mother\'s qualification': 1.0,
    'Debtor': 0.0,
    'Curricular units 1st sem (evaluations)': 13.0,
    'Gender': 1.0,
    'Curricular units 2nd sem (without evaluations)': 0.0,
    'Admission grade': 137.3,
    'Curricular units 2nd sem (enrolled)': 11.0,
    'Tuition fees up to date': 1.0
}


N_SAMPLES = 500

# Generate synthetic data for the new feature names
np.random.seed(42)
data = {}
for name in FINAL_FEATURE_NAMES:
    # Generate random float data
    data[name] = np.random.rand(N_SAMPLES) * 100

df = pd.DataFrame(data)

# Create a synthetic target variable for demonstration
df['Target'] = np.random.choice(['Dropout', 'Graduate', 'Enrolled'], 
                                size=N_SAMPLES, p=[0.33333333, 0.33333333, 0.33333333])

# Split into training data
X = df.drop("Target", axis=1)
y = df["Target"]
X_train = X.iloc[:400]
y_train = y.iloc[:400]

# Label Encoding
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# --- Define the features used for the model ---
selected_features = FINAL_FEATURE_NAMES

# --- Model Training Continuation ---

# SMOTE for balance on selected features
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train[selected_features], y_train_encoded)

# Random Forest Classifier (The core model)
model = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
)
model.fit(X_train_res, y_train_res)

# Scaler (Fitted only on training data for the selected features)
scaler = StandardScaler()
scaler.fit(X_train[selected_features])

# --- 2. Flask Application Setup ---

app = Flask(__name__)

# HTML Template Structure (index.html content, embedded here for execution)
# Note: Tailwind CSS and JavaScript are included directly in this string.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Academic Success Prediction</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #f7fafc;
        }}
    </style>
</head>
<body class="p-4 sm:p-8 flex justify-center min-h-screen">
    <div class="bg-white p-6 sm:p-10 rounded-xl shadow-2xl w-full max-w-2xl mt-10">
        <h2 class="text-3xl font-bold text-gray-800 mb-6 border-b pb-3 text-center">Student Outcome Prediction (NB-RF)</h2>
        <p class="text-sm text-gray-600 mb-6 text-center">
            Enter numerical values for the 20 academic and demographic features below to predict the student outcome. Default values from your last request are pre-filled.
        </p>
        
        <form id="predictionForm" class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {feature_inputs}
            <div class="sm:col-span-2 pt-4">
                <button type="submit" 
                        class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg shadow-md transition duration-300 transform hover:scale-[1.005] focus:outline-none focus:ring-4 focus:ring-blue-300">
                    Predict Outcome
                </button>
            </div>
        </form>

        <div id="resultContainer" class="mt-8 p-6 bg-gray-100 rounded-xl shadow-inner hidden border-l-4 border-blue-500">
            <h3 class="text-xl font-semibold text-gray-700 mb-2">Prediction Result:</h3>
            <p id="resultText" class="text-3xl font-extrabold text-blue-600"></p>
            <p id="errorText" class="text-sm text-red-500 mt-2"></p>
        </div>
    </div>

    <script>
        const form = document.getElementById('predictionForm');
        const resultContainer = document.getElementById('resultContainer');
        const resultText = document.getElementById('resultText');
        const errorText = document.getElementById('errorText');

        form.addEventListener('submit', async function(e) {{
            e.preventDefault();
            
            resultText.textContent = 'Calculating...';
            errorText.textContent = '';
            resultContainer.classList.remove('hidden');
            
            const formData = new FormData(form);
            const data = Object.fromEntries(formData.entries());
            
            // Convert all input values to numbers/floats
            const payload = {{}};
            for (const key in data) {{
                try {{
                    // Use parseFloat to handle decimal input
                    payload[key] = parseFloat(data[key]);
                }} catch (e) {{
                    // Handle non-numeric input gracefully
                    resultText.textContent = 'Invalid Input';
                    errorText.textContent = `Please enter a valid number for ${{key}}.`;
                    resultContainer.classList.remove('hidden');
                    return;
                }}
            }}


            try {{
                const response = await fetch('/predict', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(payload)
                }});

                const result = await response.json();

                if (response.ok) {{
                    const label = result.label;
                    let color = 'text-blue-600';
                    if (label === 'Dropout') {{
                        color = 'text-red-600';
                    }} else if (label === 'Enrolled') {{
                        color = 'text-yellow-600';
                    }} else if (label === 'Graduate') {{
                        color = 'text-green-600';
                    }}

                    resultText.className = `text-3xl font-extrabold ${{color}}`;
                    resultText.textContent = `Predicted Outcome: ${{label}}`;
                    errorText.textContent = `(Model Code: ${{result.prediction}})`;
                }} else {{
                    resultText.textContent = 'Prediction Failed';
                    errorText.textContent = `Error: ${{result.error || 'Unknown error'}}`;
                    resultText.className = 'text-3xl font-extrabold text-red-600';
                }}

            }} catch (error) {{
                resultText.textContent = 'Network Error';
                errorText.textContent = `Could not connect to service: ${{error.message}}`;
                resultText.className = 'text-3xl font-extrabold text-red-600';
            }}
        }});
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Renders the HTML form dynamically inserting the selected feature inputs."""
    
    # Generate the HTML input fields for the selected features
    input_fields = ""
    for feature in selected_features:
        # Use the fixed default value or 0.0 if not found
        default_value = DEFAULT_VALUES.get(feature, 0.0) 
        
        # Format the value to two decimal places for display
        formatted_value = f"{default_value:.2f}"

        input_fields += f"""
        <div class="flex flex-col">
            <label for="{feature}" class="text-sm font-medium text-gray-700 mb-1 truncate" title="{feature}">{feature}</label>
            <input type="number" step="any" id="{feature}" name="{feature}" value="{formatted_value}" 
                   class="p-3 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500" 
                   placeholder="Enter value" required>
        </div>
        """
    
    # Inject the inputs into the template and return, simulating render_template
    return HTML_TEMPLATE.format(feature_inputs=input_fields)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests using the in-memory trained model."""
    try:
        # Extract data from the request payload
        input_data = request.get_json()

        # Create a DataFrame from the input data
        # Ensure correct order and data types, using only selected features
        input_values = [float(input_data.get(f, 0.0)) for f in selected_features]
        df = pd.DataFrame([input_values], columns=selected_features)

        # Scale the data using the fitted scaler
        df_scaled = scaler.transform(df)

        # Predict
        pred = model.predict(df_scaled)
        
        # Inverse transform the prediction to get the original class label
        class_label = le.inverse_transform(pred)[0]

        return jsonify({
            "prediction": int(pred[0]),
            "label": class_label
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        # Return a user-friendly error message
        return jsonify({"error": f"Error processing input data: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(debug=True)
