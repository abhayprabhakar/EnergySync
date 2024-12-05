import os
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from dotenv import load_dotenv

# Import the PowerConsumptionAssistant from the separate module
from model import PowerConsumptionAssistant

# Load environment variables
load_dotenv()

# Flask Application Setup
app = Flask(__name__)
CORS(app)

# Initialize the assistant
MODEL_PATH = './results/power_consumption_model.pkl'
assistant = PowerConsumptionAssistant(
    csv_path=r"RealTimeData\power_consumption_10min_averages.csv", 
    model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else None
)

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    # Generate power consumption response
    response = assistant.generate_power_consumption_response(user_message)
    
    return jsonify({"response": response})

@app.route('/power_data')
def power_data():
    """
    Endpoint to retrieve power consumption data for chart rendering
    """
    return jsonify(assistant.get_power_consumption_data())

if __name__ == '__main__':
    app.run(debug=False)
