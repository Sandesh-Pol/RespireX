import numpy as np
import pickle
from keras.models import load_model

class Level0:
    def __init__(self, model_path='model/lung_cancer_model.h5', scaler_path='model/scaler.pkl'):
        try:
            self.model = load_model(model_path)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        except Exception as e:
            raise Exception(f"Error loading model or scaler: {e}")
    
    def predict(self, input_data):
        try:
            # Ensure input is a numpy array and reshape if necessary
            input_data = np.array(input_data).reshape(1, -1)
            
            # Scale input data
            input_data_scaled = self.scaler.transform(input_data)
            
            # Predict probability
            prediction = self.model.predict(input_data_scaled)
            
            # Convert probability to binary class (0 or 1)
            return int(prediction[0][0] > 0.5)
        
        except Exception as e:
            return f"Prediction error: {e}"

# Example usage
if __name__ == "__main__":
    model = Level0()
    test_data = [0, 55, 2, 2, 1, 2, 2, 2, 1, 2,2, 2, 1, 2, 2]  # Sample input
    result = model.predict(test_data)
    print("Predicted Lung Cancer Risk:", result)
