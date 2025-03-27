import numpy as np
import joblib
import tensorflow as tf

def load_model():
    return tf.keras.models.load_model('lung_cancer_model.h5')

def load_scaler():
    return joblib.load('scaler.pkl')

def predict_lung_cancer(symptoms):
    model = load_model()
    scaler = load_scaler()
    
    symptoms_array = np.array(symptoms).reshape(1, -1)
    symptoms_scaled = scaler.transform(symptoms_array)
    
    prediction = model.predict(symptoms_scaled) * 100  
    return f'Lung Cancer Risk: {prediction[0][0]:.2f}%'

if __name__ == "__main__":
    print("Enter the following details:")

    gender = int(input("Gender (1: Male, 2: Female): ")) 
    age = int(input("Age: "))
    smoking = int(input("Smoking (1: No, 2: Yes): ")) 
    yellow_fingers = int(input("Yellow Fingers (1: No, 2: Yes): ")) 
    anxiety = int(input("Anxiety (1: No, 2: Yes): ")) 
    peer_pressure = int(input("Peer Pressure (1: No, 2: Yes): ")) 
    chronic_disease = int(input("Chronic Disease (1: No, 2: Yes): ")) 
    fatigue = int(input("Fatigue (1: No, 2: Yes): ")) 
    allergy = int(input("Allergy (1: No, 2: Yes): ")) 
    wheezing = int(input("Wheezing (1: No, 2: Yes): ")) 
    alcohol_consuming = int(input("Alcohol Consuming (1: No, 2: Yes): ")) 
    coughing = int(input("Coughing (1: No, 2: Yes): ")) 
    shortness_of_breath = int(input("Shortness of Breath (1: No, 2: Yes): ")) 
    swallowing_difficulty = int(input("Swallowing Difficulty (1: No, 2: Yes): ")) 
    chest_pain = int(input("Chest Pain (1: No, 2: Yes): ")) 

    user_input = [
        gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
        chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
        coughing, shortness_of_breath, swallowing_difficulty, chest_pain
    ]

    result = predict_lung_cancer(user_input)
    print(result)
