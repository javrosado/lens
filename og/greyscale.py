import cv2

image =cv2.imread("./pattern/paper.jpg")

grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
new_width = int(23)
new_height = int(23)
resized_image = cv2.resize(grey_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
inverted_image = 255 - grey_image
cv2.imwrite("./pattern/pattern1reduced.jpg" , resized_image)
cv2.imwrite("./pattern/pattern1grey.jpg" , grey_image)
cv2.imwrite("./pattern/pattern1inverted.jpg" , inverted_image)
