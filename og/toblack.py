from PIL import Image

# Load the image
image_path = './pattern/airforce.jpg'
image = Image.open(image_path)

# Convert the image to RGB mode if it's not already
image = image.convert('L')

# Get the pixel data
pixels = image.load()

# Define white color
white = 200

# Process each pixel
for y in range(image.height):
    for x in range(image.width):
        if pixels[x, y] < white:
            pixels[x, y] = 0
        else: pixels[x,y] = 255 

# Save the edited image
edited_image_path = 'edited_image.jpg'
image.save(edited_image_path)