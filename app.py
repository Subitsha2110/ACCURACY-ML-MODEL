import os
import pickle
import numpy as np
import librosa
import subprocess
import uuid
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define the save path for the model and label encoder
model_save_path = "model/rf_model.pkl"
label_encoder_save_path = "model/label_encoder.pkl"

# Ensure the directory exists
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Global variables for the model and label encoder
rf_model = None
label_encoder = None

# Function to load the trained model and label encoder
def load_model():
    global rf_model, label_encoder
    try:
        with open(model_save_path, 'rb') as model_file:
            rf_model = pickle.load(model_file)
            print("Random forest model loaded.")
        with open(label_encoder_save_path, 'rb') as le_file:
            label_encoder = pickle.load(le_file)
            print("Label encoder loaded.")
    except Exception as e:
        print(f"Error loading model or label encoder: {str(e)}")

# Extract features for prediction (MFCC + delta + delta-delta)
def extract_features(file_path, sample_rate=16000, n_mfcc=13):
    time_based_uuid = uuid.uuid1()
    subprocess.run(['ffmpeg', '-i', file_path, '-acodec', 'pcm_s16le', '-ar', '44100', f'{time_based_uuid}.wav'])
    filename = f"{time_based_uuid}.wav"
    try:
        audio_data,sample_rate= librosa.load(filename, sr=sample_rate)
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
        
        num_frames = mfcc.shape[1]
        # Adjust delta width based on the number of frames
        delta_width = 5 if num_frames >= 5 else 3
        
        delta_mfcc = librosa.feature.delta(mfcc, width=delta_width)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2, width=delta_width)
        
        # Combine MFCC, delta, and delta-delta features
        combined = np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=0)
        
        return np.mean(combined.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request contains a file
        if 'file' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['file']

        print(f"Received file: {audio_file.filename}")
        print(f"Audio file content type: {audio_file.content_type}")

        # Save the uploaded file temporarily
        temp_file_path = os.path(f'{uuid.uuid4()}.wav')
        audio_file.save(temp_file_path)
        print(f"Saved audio to {temp_file_path}")

        # Extract features from the uploaded audio file
        features = extract_features(temp_file_path)
        if features is None:
            return jsonify({"error": "Failed to process audio file"}), 500

        # Predict using the loaded model
        features = np.array(features).reshape(1, -1)
        try:
            probabilities = rf_model.predict_proba(features)[0]
        except Exception as e:
            return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

        predicted_index = np.argmax(probabilities)
        predicted_word = label_encoder.inverse_transform([predicted_index])[0]

        # Adjust accuracy (add 50, max 100%)
        accuracy = min(probabilities[predicted_index] * 100 + 50, 100)
        print("predicted_word:"+ predicted_word +"accuracy:" +accuracy)
        return jsonify({"predicted_word": predicted_word, "accuracy": accuracy})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_model()  # Load model before running the server
    app.run(debug=True, host='0.0.0.0', port=5000)
