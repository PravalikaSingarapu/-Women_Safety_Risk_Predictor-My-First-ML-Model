from flask import Flask, render_template, request
import pickle
import re
import numpy as np
from scipy.sparse import hstack

app = Flask(__name__)

# Load model, vectorizer, encoders (will be created after training)
model = None
vectorizer = None
label_encoders = None
target_le = None

# Try to load the model if it exists
try:
    with open('model/safety_model.pkl', 'rb') as f:
        model, vectorizer, label_encoders, target_le = pickle.load(f)
except FileNotFoundError:
    print("Model not found. Please run train_model.py first.")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Safety Tips and Emergency Actions
safety_tips = {
    'Safe': "You seem to be in a safe environment. Stay alert and keep your phone charged.",
    'Caution': "Be cautious. Share your location with trusted contacts and avoid isolated areas.",
    'Danger': "IMMEDIATE ACTION REQUIRED! You're in a potentially dangerous situation."
}

# Detailed emergency suggestions
emergency_actions = {
    'Safe': [
        "✅ Keep your phone charged above 50%",
        "📍 Share your location with family/friends",
        "👀 Stay aware of your surroundings",
        "🚶‍♀️ Walk confidently and purposefully"
    ],
    'Caution': [
        "📱 Call/text someone you trust immediately",
        "🏃‍♀️ Move to a well-lit, crowded area",
        "📍 Share live location with emergency contacts",
        "🚨 Have emergency numbers ready to dial",
        "👥 Stay near other people, avoid being alone",
        "🔦 Use phone flashlight if dark"
    ],
    'Danger': [
        "🚨 CALL 911 or local emergency number NOW",
        "📞 Speed dial trusted emergency contact",
        "🏃‍♀️ Get to nearest safe public place immediately",
        "📢 Don't hesitate to shout for help if needed",
        "🏪 Enter nearest open store/building",
        "🚕 Call a ride-share to safe location",
        "📱 Use emergency SOS feature on your phone",
        "👮‍♀️ Find police officer or security guard",
        "🔊 Make noise to attract attention",
        "📍 Send live location to multiple contacts"
    ]
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded. Please train the model first."
    
    # Get form data
    time_of_day = request.form['time_of_day']
    location_type = request.form['location_type']
    crowd_density = request.form['crowd_density']
    is_alone = request.form['is_alone']
    battery_level = int(request.form['battery_level'])
    mood_text = request.form['mood_text']

    # Encode categorical values
    try:
        encoded = [
            label_encoders['time_of_day'].transform([time_of_day])[0],
            label_encoders['location_type'].transform([location_type])[0],
            label_encoders['crowd_density'].transform([crowd_density])[0],
            label_encoders['is_alone'].transform([is_alone])[0],
            battery_level
        ]

        # Clean and vectorize text
        cleaned_text = clean_text(mood_text)
        text_vector = vectorizer.transform([cleaned_text])

        # Combine text vector and numerical features
        numeric_features = np.array(encoded).reshape(1, -1)
        combined_features = hstack([text_vector, numeric_features])

        # Make prediction
        prediction = model.predict(combined_features)[0]
        risk_level = target_le.inverse_transform([prediction])[0]

        # Get safety tip and emergency actions
        tip = safety_tips.get(risk_level, "Stay safe!")
        actions = emergency_actions.get(risk_level, [])

        return render_template('result.html', 
                             risk=risk_level, 
                             tip=tip, 
                             actions=actions)
    
    except Exception as e:
        return f"Error making prediction: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
