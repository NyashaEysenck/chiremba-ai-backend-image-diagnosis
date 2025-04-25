from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
import matplotlib
# Force matplotlib to use TkAgg backend for display
matplotlib.use('TkAgg')

# Import the Gemini interpretation module
try:
    from gemini_interpretation import get_gemini_interpretation
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Gemini interpretation module not available. Using rule-based interpretation.")

# Define image size
IMAGE_SIZE = (350, 350)

# Define class labels
class_labels = ['adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
 'normal',
 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa']

def load_and_preprocess_image(img_path, target_size):
    """Load and preprocess an image for prediction."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image like the training images
    return img_array

def initialize_model():
    """Initialize and load the lung cancer prediction model."""
    # Check if the complete model exists, if so, load it directly
    if os.path.exists("lung_cancer_prediction_model_complete.h5"):
        print("Loading complete model...")
        model = keras.models.load_model("lung_cancer_prediction_model_complete.h5")
        return model
    
    # If complete model doesn't exist, recreate it from weights
    print("Complete model not found. Recreating model from weights...")
    OUTPUT_SIZE = 4  # Number of classes
    pretrained_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
    pretrained_model.trainable = False

    model = keras.models.Sequential()
    model.add(pretrained_model)
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(OUTPUT_SIZE, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load only the weights from the HDF5 file
    model.load_weights("best_model.hdf5")
    
    # Save the complete model for future use
    model.save("lung_cancer_prediction_model_complete.h5")
    print(f"Complete model saved as: {os.path.abspath('lung_cancer_prediction_model_complete.h5')}")
    
    return model

def generate_gradcam(model, img_array, pred_index=None):
    """
    Generate a saliency map visualization for the model.
    This implementation doesn't rely on accessing model layers and works with any model.
    
    Args:
        model: The trained model
        img_array: Preprocessed image array
        pred_index: Index of the class to generate visualization for (default: highest scoring class)
        
    Returns:
        heatmap: Saliency heatmap
        superimposed_img: Original image with heatmap overlay
    """
    # Make a prediction to determine the class index if not provided
    preds = model.predict(img_array)
    if pred_index is None:
        pred_index = np.argmax(preds[0])
    
    # Create a copy of the input image that requires gradient computation
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        target_class = predictions[:, pred_index]
    
    # Get gradients of the target class with respect to the input image
    gradients = tape.gradient(target_class, img_tensor)
    
    # Take the maximum gradient across RGB channels
    gradients = tf.reduce_max(tf.abs(gradients), axis=-1)
    
    # Normalize gradients to range [0, 1]
    gradients = (gradients - tf.reduce_min(gradients)) / (tf.reduce_max(gradients) - tf.reduce_min(gradients) + tf.keras.backend.epsilon())
    
    # Convert to numpy array
    heatmap = gradients.numpy()[0]
    
    # Resize if needed
    if heatmap.shape != (img_array.shape[1], img_array.shape[2]):
        heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    
    # Convert heatmap to RGB using the jet colormap
    heatmap_rgb = np.uint8(255 * heatmap)
    heatmap_rgb = cv2.applyColorMap(heatmap_rgb, cv2.COLORMAP_JET)
    
    # Convert from BGR to RGB (OpenCV uses BGR by default)
    heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
    
    # Superimpose the heatmap on original image
    orig_img = (img_array[0] * 255).astype(np.uint8)
    superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap_rgb, 0.4, 0)
    
    # Convert to PIL Image for compatibility with the rest of the code
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    return heatmap, superimposed_img

def get_clinical_interpretation(predicted_label, confidence):
    """
    Generate a clinical interpretation based on the predicted class and confidence.
    
    Args:
        predicted_label: The predicted class label
        confidence: The confidence score for the prediction
        
    Returns:
        str: Clinical interpretation text
    """
    interpretations = {
        'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib': {
            'description': 'Adenocarcinoma in the left lower lobe',
            'stage': 'Stage Ib (T2, N0, M0)',
            'characteristics': 'Primary tumor > 3cm but ≤ 5cm, no regional lymph node metastasis, no distant metastasis',
            'recommendations': [
                'Surgical resection is typically the primary treatment',
                'Consider adjuvant chemotherapy based on risk factors',
                'Regular follow-up imaging to monitor for recurrence'
            ]
        },
        'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa': {
            'description': 'Large cell carcinoma in the left hilum',
            'stage': 'Stage IIIa (T2, N2, M0)',
            'characteristics': 'Primary tumor > 3cm but ≤ 5cm, metastasis in ipsilateral mediastinal lymph nodes, no distant metastasis',
            'recommendations': [
                'Multimodality treatment approach often recommended',
                'Combination of chemotherapy and radiation therapy',
                'Surgical resection may be considered in selected cases',
                'Immunotherapy may be appropriate based on biomarker testing'
            ]
        },
        'normal': {
            'description': 'No evidence of lung cancer',
            'recommendations': [
                'Continue routine screening based on risk factors',
                'Maintain healthy lifestyle habits',
                'Follow up as clinically indicated'
            ]
        },
        'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': {
            'description': 'Squamous cell carcinoma in the left hilum',
            'stage': 'Stage IIIa (T1, N2, M0)',
            'characteristics': 'Primary tumor ≤ 3cm, metastasis in ipsilateral mediastinal lymph nodes, no distant metastasis',
            'recommendations': [
                'Multimodality treatment approach often recommended',
                'Combination of chemotherapy and radiation therapy',
                'Surgical resection may be considered in selected cases',
                'Consider targeted therapy or immunotherapy based on biomarker testing'
            ]
        }
    }
    
    confidence_level = "high" if confidence > 0.9 else "moderate" if confidence > 0.7 else "low"
    
    if predicted_label in interpretations:
        info = interpretations[predicted_label]
        
        if predicted_label == 'normal':
            interpretation = f"The scan appears normal with {confidence_level} confidence ({confidence*100:.1f}%).\n\n"
            interpretation += f"Description: {info['description']}\n\n"
            interpretation += "Recommendations:\n"
            for rec in info['recommendations']:
                interpretation += f"- {rec}\n"
        else:
            interpretation = f"The scan suggests {info['description']} with {confidence_level} confidence ({confidence*100:.1f}%).\n\n"
            interpretation += f"Stage: {info['stage']}\n"
            interpretation += f"Characteristics: {info['characteristics']}\n\n"
            interpretation += "Recommendations:\n"
            for rec in info['recommendations']:
                interpretation += f"- {rec}\n"
            
            if confidence < 0.7:
                interpretation += "\nNote: Due to the lower confidence level, additional diagnostic procedures may be warranted to confirm this finding."
    else:
        interpretation = "No specific interpretation available for this prediction."
    
    return interpretation

def predict_with_visualization(image_path, use_gemini=False):
    """
    Predict the class of an image and provide visualization and interpretation.
    
    Args:
        image_path: Path to the image file
        use_gemini: Whether to use Gemini AI for interpretation (if available)
        
    Returns:
        dict: Dictionary containing prediction results, visualization, and interpretation
    """
    # Check if the image file exists
    if not os.path.exists(image_path):
        return {
            "error": f"Image file not found: {image_path}"
        }
    
    # Load and preprocess the image
    img = load_and_preprocess_image(image_path, IMAGE_SIZE)
    
    # Load the model
    model = initialize_model()
    
    # Make prediction
    start_time = time.time()
    preds = model.predict(img)
    prediction_time = time.time() - start_time
    
    # Get the predicted class index and label
    pred_index = np.argmax(preds[0])
    predicted_label = class_labels[pred_index]
    
    # Get confidence score
    confidence = preds[0][pred_index]
    
    # Get the second highest prediction
    preds_copy = preds[0].copy()
    preds_copy[pred_index] = -1  # Set the highest to -1
    second_pred_index = np.argmax(preds_copy)
    second_predicted_label = class_labels[second_pred_index]
    second_confidence = preds[0][second_pred_index]
    
    # Generate Grad-CAM visualization
    try:
        start_time = time.time()
        heatmap, superimposed_img = generate_gradcam(model, img, pred_index)
        gradcam_time = time.time() - start_time
    except Exception as e:
        print(f"Error generating Grad-CAM visualization: {str(e)}")
        heatmap, superimposed_img = None, None
        gradcam_time = 0
    
    # Get interpretation
    if use_gemini and GEMINI_AVAILABLE:
        # Load the original image for Gemini
        original_img = Image.open(image_path).resize((350, 350))
        
        # Get Gemini interpretation
        try:
            interpretation = get_gemini_interpretation(
                predicted_label, 
                confidence, 
                original_img, 
                superimposed_img
            )
        except Exception as e:
            print(f"Error with Gemini interpretation: {str(e)}")
            # Fall back to rule-based interpretation
            interpretation = get_clinical_interpretation(predicted_label, confidence)
    else:
        # Use rule-based interpretation
        interpretation = get_clinical_interpretation(predicted_label, confidence)
    
    # Prepare the result dictionary
    result = {
        "predicted_class": predicted_label,
        "confidence": float(confidence),
        "second_prediction": {
            "class": second_predicted_label,
            "confidence": float(second_confidence)
        },
        "all_probabilities": {class_labels[i]: float(preds[0][i]) for i in range(len(class_labels))},
        "interpretation": interpretation,
        "processing_times": {
            "prediction": prediction_time,
            "gradcam": gradcam_time
        }
    }
    
    # Add visualization to the result
    if superimposed_img:
        result["has_visualization"] = True
        result["superimposed_img"] = superimposed_img
    else:
        result["has_visualization"] = False
    
    return result

# If this script is run directly, perform a prediction on a sample image
if __name__ == "__main__":
    import sys
    import argparse
    import time
    from PIL import Image
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Lung Cancer Prediction with Visualization')
    parser.add_argument('--image', type=str, help='Path to the image file')
    parser.add_argument('--use-gemini', action='store_true', help='Use Gemini AI for interpretation')
    args = parser.parse_args()
    
    # Get image path from arguments or use default
    if args.image:
        image_path = args.image
    else:
        # Use a default image with absolute path
        image_path = os.path.join(os.getcwd(), "dataset", "test", "adenocarcinoma", "000114.png")
        print(f"Using default image: {image_path}")
        # Verify the image exists
        if not os.path.exists(image_path):
            print(f"Error: Default image not found at {image_path}")
            # Try to find any PNG file in the dataset
            for root, dirs, files in os.walk(os.path.join(os.getcwd(), "dataset")):
                for file in files:
                    if file.endswith('.png'):
                        image_path = os.path.join(root, file)
                        print(f"Found alternative image: {image_path}")
                        break
                if os.path.exists(image_path):
                    break
    
    # Perform prediction with visualization
    result = predict_with_visualization(image_path, use_gemini=args.use_gemini)
    
    # Print the results
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"\nPredicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"Second Most Likely: {result['second_prediction']['class']} ({result['second_prediction']['confidence']*100:.2f}%)")
        print("\nAll Class Probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob*100:.2f}%")
        
        print("\nInterpretation:")
        print(result['interpretation'])
        
        print(f"\nProcessing Times:")
        print(f"  Prediction: {result['processing_times']['prediction']:.3f} seconds")
        print(f"  Grad-CAM: {result['processing_times']['gradcam']:.3f} seconds")
        
        # Show the original image and Grad-CAM visualization if available
        if result["has_visualization"]:
            # Load and display the original image
            original_img = load_and_preprocess_image(image_path, target_size=(350, 350))
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(original_img[0])
            plt.axis('off')
            
            # Display the Grad-CAM visualization
            plt.subplot(1, 2, 2)
            plt.title("Grad-CAM Visualization")
            plt.imshow(result["superimposed_img"])
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()