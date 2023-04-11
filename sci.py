import sys
from PIL import Image

image_path = sys.argv[1]
try:
    image = Image.open(image_path)
    print("Image opened")
    print(image)
    # Use the image variable for further processing
except IOError:
    print(f"Error opening image at {image_path}")
