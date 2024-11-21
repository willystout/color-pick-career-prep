# from flask import Flask, request, jsonify
# from flask_cors import CORS  
# import numpy as np
# # import joblib
# import os
# import torch
# import torch.nn as nn
# import pickle

# app = Flask(__name__)
# CORS(app)  # Enable CORS

# # Define the neural network
# class NeuralNet(nn.Module):
#     def __init__(self):
#         super(NeuralNet, self).__init__()
#         self.fc1 = nn.Linear(3, 32)
#         self.fc2 = nn.Linear(32, 16)
#         self.fc3 = nn.Linear(16, 10)  # 10 classes

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.softmax(self.fc3(x), dim=1)
#         return x

# # Add error handling for model loading
# try:
#     # knn_model = joblib.load('/Users/williamstout/Dropbox/My Mac (William''s MacBook Pro (2))/Desktop/career_prep/color-picker/api/knn_model.joblib')
#     # Load the entire NN model
#     nn = torch.load("./nn_model.pth")
#     nn.eval()  # Set to evaluation mode
#     print("NN model loaded successfully!")

#     # Load the label encoder
#     with open("./label_encoder.pkl", "rb") as f:
#         label_encoder = pickle.load(f)
#     print("Label encoder loaded successfully!")
    
# except Exception as e:
#     print(f"Error loading model: {e}")
#     nn = None

# @app.route("/api/identify", methods=['GET'])
# def classify_color():
#     def get_label(hex_input, model, label):
#         if nn is None:
#             return jsonify({"error": "Model not loaded"}), 500
#         if not hex_input:
#             return jsonify({"error": "No color parameter provided"}), 400
#         sample_rgb = np.array([hex_to_rgb(hex_input)], dtype=np.float32)
#         rgb_tensor = torch.tensor(sample_rgb, dtype=torch.float32)
#         predicted_label = model(rgb_tensor)
#         predicted_class = torch.argmax(predicted_label, dim=1).item()
#         predicted_name = label.inverse_transform([predicted_class])
#         print("PREDICTED NAME: ", predicted_name)
#         return predicted_name
    
#     def hex_to_rgb(hex_color):
#         hex_color = hex_color.lstrip('#')
#         return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
#     # Get the color parameter from the URL
#     hex_color = request.args.get('color')
#     try:
#         # Convert the hex color to RGB
#         # rgb_color = np.array([hex_to_rgb(hex_color)])
#         # Predict the color label using the KNN model
#         predicted_label = get_label(hex_color, nn, label_encoder)
#         # Return the result as JSON
#         print(predicted_label)
#         print(hex_color)
#         return {"color": hex_color, "label": predicted_label}
#     except ValueError as e:
#         return jsonify({"error": f"Invalid color format: {str(e)}"}), 400
#     except Exception as e:
#         return jsonify({"error": f"Unexpected error: {str(e)}"}), 500



# # def hex_to_rgb(hex_color):
# #     hex_color = hex_color.lstrip('#')
# #     if len(hex_color) != 6:
# #         raise ValueError("Color must be 6 hex digits")
# #     return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# # Add a root route for testing
# @app.route("/")
# def home():
#     return jsonify({"status": "API is running"})

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, request, jsonify
from flask_cors import CORS  
import numpy as np
import torch
import torch.nn as nn
import pickle

app = Flask(__name__)
CORS(app)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)  # 10 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

# Add error handling for model loading
try:
    nn = torch.load("./nn_model.pth")
    nn.eval()  # Set to evaluation mode
    print("NN model loaded successfully!")

    with open("./label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    print("Label encoder loaded successfully!")
    
except Exception as e:
    print(f"Error loading model: {e}")
    nn = None

@app.route("/api/identify", methods=['GET'])
def classify_color():
    def get_label(hex_input, model, label):
        if model is None:
            raise ValueError("Model not loaded")
        if not hex_input:
            raise ValueError("No color parameter provided")
        
        sample_rgb = np.array([hex_to_rgb(hex_input)], dtype=np.float32)
        rgb_tensor = torch.tensor(sample_rgb, dtype=torch.float32)
        with torch.no_grad():  # Add this for inference
            predicted_label = model(rgb_tensor)
        predicted_class = torch.argmax(predicted_label, dim=1).item()
        predicted_name = label.inverse_transform([predicted_class])[0]  # Get first item
        return predicted_name
    
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            raise ValueError("Color must be 6 hex digits")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    # Get the color parameter from the URL
    hex_color = request.args.get('color')
    
    try:
        predicted_label = get_label(hex_color, nn, label_encoder)
        print(f"Predicted label: {predicted_label}")
        print(f"Hex color: {hex_color}")
        return predicted_label
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/")
def home():
    return jsonify({"status": "API is running"})

if __name__ == '__main__':
    app.run(debug=True)