import tensorflow as tf
from tensorflow.keras.models import load_model
import glob
import imageio
import numpy as np
import os

# Define the PSNR calculation function
def calculate_psnr(true_image, predicted_image):
    return tf.image.psnr(true_image, predicted_image, max_val=1.0)

# Load the pre-trained model with a custom PSNR metric
model = load_model('FINAL_MODEL.h5', custom_objects={'psnr_metric': calculate_psnr})

# Function to enhance an image recursively
def enhance_image(image, iterations, is_first_pass):
    if iterations == 0:
        return image

    height, width, channels = image.shape
    prediction = model.predict(image.reshape(1, height, width, channels))
    
    if is_first_pass:
        normalized_image = image / 255.0
        enhanced_image = normalized_image + ((prediction[0] * normalized_image) * (1 - normalized_image))
        psnr_value = calculate_psnr(tf.convert_to_tensor(image, dtype=tf.float32), 
                                    tf.convert_to_tensor(enhanced_image * 255, dtype=tf.float32)).numpy()
        print(f"PSNR: {psnr_value:.4f}")
        return enhance_image(enhanced_image, iterations - 1, False)
    else:
        enhanced_image = image + ((prediction[0] * image) * (1 - image))
        psnr_value = calculate_psnr(tf.convert_to_tensor(image, dtype=tf.float32), 
                                    tf.convert_to_tensor(enhanced_image, dtype=tf.float32)).numpy()
        print(f"PSNR: {psnr_value:.4f}")
        return enhance_image(enhanced_image, iterations - 1, False)

# Load and process images from the specified directory
input_dir = 'test/low'
image_files = sorted(glob.glob(input_dir + "/*"))
images = [imageio.imread(file) for file in image_files]
images_array = np.array(images)

# Enhance the first image in the array
first_image = images_array[0]
enhanced_image_result = enhance_image(first_image, 8, True)

# Define the output directory and save the enhanced image
output_dir = 'test/predicted'
os.makedirs(output_dir, exist_ok=True)

output_filepath = os.path.join(output_dir, 'enhanced_image.png')
imageio.imwrite(output_filepath, (enhanced_image_result * 255).astype(np.uint8))
print(f"Enhanced image saved to {output_filepath}")
