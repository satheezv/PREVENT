import cv2
import os

def resize_image(input_path, output_path):
    # Read the image
    image = cv2.imread(input_path)
    
    if image is None:
        print(f"Error: Unable to read image {input_path}.")
        return
    
    # Resize the image to 640x480
    resized_image = cv2.resize(image, (640, 480))
    
    # Save the resized image
    cv2.imwrite(output_path, resized_image)
    print(f"Image saved at {output_path}")

def process_folder(input_folder, output_folder):
    for root, _, files in os.walk(input_folder):
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        
        for filename in files:
            input_path = os.path.join(root, filename)
            output_path = os.path.join(output_subfolder, filename)
            
            if os.path.isfile(input_path):
                resize_image(input_path, output_path)

# Example usage
input_folder = "W:\gasSensor_ws\GasSensor_ws\data_v2\\test_images"  # Change to your folder path
output_folder = "W:\gasSensor_ws\GasSensor_ws\data_v2\\test_images\\reduced_size"  # Change to your desired output folder
process_folder(input_folder, output_folder)
