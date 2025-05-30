import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

class PlantDiseaseDetector:
    def __init__(self, model_path='plant_disease_final_model.keras'):
        """
        Initialize the plant disease detector with a trained model.
        
        Args:
            model_path (str): Path to the saved Keras model file
        """
        self.IMAGE_SIZE = 224
        self.model = load_model(model_path)
        
        # Define class labels for plant diseases
        self.class_labels = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]

    def preprocess_image(self, image_path):
        """
        Preprocess an image for prediction.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        try:
            img = image.load_img(image_path, target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize
            return img_array
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")

    def predict(self, image_path):
        """
        Predict plant disease from an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (disease_name, confidence, advice)
        """
        try:
            # Preprocess the image
            img_array = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Get the predicted disease
            disease = self.class_labels[predicted_class]
            
            # Generate advice based on the disease
            advice = self.generate_advice(disease)
            
            return disease, confidence, advice
            
        except Exception as e:
            raise ValueError(f"Error during prediction: {str(e)}")

    def generate_advice(self, disease):
        """Generate advice based on the detected disease"""
        advice_dict = {
            'Apple___Apple_scab': 'Apply fungicide sprays in early spring. Prune infected branches and maintain good air circulation.',
            'Apple___Black_rot': 'Remove infected fruit and branches. Apply fungicide during wet weather periods.',
            'Apple___Cedar_apple_rust': 'Remove nearby cedar trees if possible. Apply fungicide in early spring.',
            'Corn_(maize)___Common_rust_': 'Use resistant varieties. Apply fungicide when symptoms first appear.',
            'Corn_(maize)___Northern_Leaf_Blight': 'Rotate crops and use resistant varieties. Apply fungicide if needed.',
            'Grape___Black_rot': 'Prune vines to improve air circulation. Apply fungicide before and after bloom.',
            'Potato___Early_blight': 'Rotate crops and use certified seed. Apply fungicide preventatively.',
            'Potato___Late_blight': 'Use resistant varieties. Apply fungicide before symptoms appear.',
            'Tomato___Bacterial_spot': 'Use disease-free seed. Avoid overhead irrigation. Apply copper-based bactericides.',
            'Tomato___Early_blight': 'Rotate crops and use resistant varieties. Apply fungicide preventatively.',
            'Tomato___Late_blight': 'Use resistant varieties. Apply fungicide before symptoms appear.',
            'Tomato___Leaf_Mold': 'Improve air circulation. Apply fungicide when conditions are favorable for disease.',
            'Tomato___Septoria_leaf_spot': 'Remove infected leaves. Apply fungicide when symptoms first appear.',
            'Tomato___Spider_mites': 'Use miticides. Maintain proper irrigation to reduce stress.',
            'Tomato___Target_Spot': 'Use resistant varieties. Apply fungicide when conditions are favorable.',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whiteflies. Use resistant varieties if available.',
            'Tomato___Tomato_mosaic_virus': 'Use virus-free seed. Control aphids. Remove infected plants.'
        }
        
        # For healthy plants
        if disease.endswith('healthy'):
            return 'Your plant appears healthy! Continue regular maintenance and monitoring.'
        
        # For known diseases
        if disease in advice_dict:
            return advice_dict[disease]
        
        # Generic advice for unknown diseases
        return 'Monitor the plant closely. Consider consulting with a local agricultural expert for specific treatment recommendations.'

def test_detector():
    """Test the detector with a sample image"""
    try:
        # Initialize detector
        detector = PlantDiseaseDetector()
        
        # Test image path (replace with a valid path to a test image)
        test_image_path = "archive (3)/test/test/PotatoHealthy2.JPG"  # Replace with your test image path
        
        if not os.path.exists(test_image_path):
            print(f"⚠️ Test image not found at: {test_image_path}")
            print("Please provide a valid path to a test image.")
            return
        
        # Make prediction
        disease, confidence, advice = detector.predict(test_image_path)
        
        # Print results
        print("\n🔍 Test Results:")
        print(f"📷 Image: {os.path.basename(test_image_path)}")
        print(f"🌿 Disease: {disease}")
        print(f"📊 Confidence: {confidence:.2%}")
        print(f"💡 Advice: {advice}")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")

if __name__ == "__main__":
    # Print available classes
    detector = PlantDiseaseDetector()
    print("\n📚 Available Plant Classes:")
    for i, class_name in enumerate(detector.class_labels):
        print(f"{i+1}. {class_name}")
    
    # Run test if a test image is available
    test_detector() 