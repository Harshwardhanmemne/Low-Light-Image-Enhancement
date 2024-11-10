## Low-Light Image Enhancement Using CNN
## Project Overview
This project focuses on enhancing the quality of images captured in low-light environments by utilizing Convolutional Neural Networks (CNNs). Images taken under low-light conditions typically exhibit significant noise, reduced visibility, and a loss of detail, which can impede numerous computer vision applications. Our goal is to develop a CNN-driven model that can successfully denoise and improve these images, thereby increasing their clarity and usefulness.

## Model Architecture
- My CNN model is designed for feature extraction and noise reduction in images. It accepts RGB images of any resolution through its input layer.
- The model includes multiple convolutional layers, each with 32 filters and 3x3 kernels. These layers use ReLU activation functions to learn complex patterns and extract important 
  features from the images.
- Skip connections are implemented to retain information from earlier layers, helping to preserve essential details that might otherwise be lost during processing. This improves the 
  model's overall performance.
- The output layer is a convolutional layer that produces the final enhanced image, applying the necessary enhancements to improve image quality.
- The detailed architecture, including specifics on each layer and their configurations, is provided in the accompanying report PDF.

## Evaluation Metric
The model's performance is evaluated using the Peak Signal-to-Noise Ratio (PSNR). The final average PSNR on the test dataset is 23.802893.

## How to Run
1. **Clone the Repository**:
    ```sh
   https://github.com/Aimank009/VLG_DenoisingImages.git
    ```
2. **Go to project directory**:
    ```sh
    cd VLG_DenoisingImages
    ```
3. **Run the Main Script**:
    ```sh
    python main.py
    ```
#### Testing Instructions
Place the low-light images you want to enhance in the test/low folder. After running the script, the denoised images will be saved in the test/predicted folder.

### Sample Result

<img width="952" alt="Screenshot 2024-06-16 at 11 29 41 PM" src="https://github.com/Aimank009/VLG_DenoisingImages/assets/128082668/e9cf1932-1705-418c-a4c8-e6da609baa96">

