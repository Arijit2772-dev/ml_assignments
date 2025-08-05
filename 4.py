import numpy as np
from PIL import Image

def img_to_array(image_path, output_txt_path):


        # 1. Open the image using Pillow
        img = Image.open(image_path)
        
        # 2. Convert the image to a NumPy array
        img_array = np.array(img)
        
        print(f"Processing '{image_path}'...")
        print(f"Image shape: {img_array.shape}")

        # 3. Check if the image is RGB or Grayscale
        if img_array.ndim == 3:
            # Case for RGB image (3 dimensions: height, width, channels)
            print("Image is RGB. Reshaping to 2D for saving.")
            h, w, c = img_array.shape
            # Reshape the 3D array into a 2D array to save it
            img_reshaped = img_array.reshape(h, w * c)
            # Save the 2D array as integers
            np.savetxt(output_txt_path, img_reshaped, fmt='%d')
            
        elif img_array.ndim == 2:
            # Case for Grayscale image (2 dimensions: height, width)
            print("Image is Grayscale. Saving directly.")
            # Save the 2D array as integers
            np.savetxt(output_txt_path, img_array, fmt='%d')
        
        else:
            print("Unsupported image format (not 2D or 3D).")
            return

        print(f"Successfully saved image data to '{output_txt_path}'")


# 1. Define the name of your image file.
#    (Make sure 'image.jpeg' is in the same folder as this script)
input_file = 'image.jpeg'

# 2. Define the name for the output text file you want to create.
output_file = 'image.txt'

# 3. Call the function, "feeding" it your input and output filenames.
img_to_array(input_file, output_file)

print("\nScript has finished running.")




