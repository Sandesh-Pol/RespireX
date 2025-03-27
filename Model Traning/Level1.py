import json
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input

class DiseasePredictor:
    def __init__(self, model_json_path, model_weights_path):
        self.labels = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
            'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
            'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
        ]
        self.model = self._load_model(model_json_path, model_weights_path)

    def _load_model(self, json_path, weights_path):
        """Load the model architecture and weights."""
        try:
            with open(json_path, 'r') as file:
                architecture = json.load(file)  # Load as dict
                model = model_from_json(json.dumps(architecture))  # Convert to JSON string
            model.load_weights(weights_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _preprocess_image(self, img_path):
        """Load and preprocess the image."""
        img = image.load_img(img_path, target_size=(128, 128))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return preprocess_input(img)

    def predict(self, img_path):
        """Predict disease probabilities from image."""
        img = self._preprocess_image(img_path)
        pred = self.model.predict(img)[0]  # Get first sample's predictions
        predictions = dict(zip(self.labels, pred))  # Map labels to probabilities
        return predictions

# Example Usage
if __name__ == "__main__":
    predictor = DiseasePredictor('model/multi_disease_model.json', 'model/team7_model.h5')
    predictions = predictor.predict('test/00018388_000.png')
    
    # Display all labels with their respective values
    for label, value in predictions.items():
        print(f"{label}: {value:.4f}")
