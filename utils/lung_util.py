
import numpy as np
import tensorflow as tf
import cv2

 
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

  

# If this script is run directly, perform a prediction on a sample image
if __name__ == "__main__":
    
    None