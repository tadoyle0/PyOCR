# Import modules
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

images = convert_from_path('example3.pdf') #Read pdf file
for i in range(len(images)):
          images[i].save('img'+str(i)+'.jpg', 'JPEG')

#Using the opencv-python library to donoise the image for 
#better handwriting recognition
img = cv.imread('img0.jpg')
dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)

cv.imwrite("img0dst.jpg", img)

# Create an image object of PIL library
image = Image.open('img0.jpg')
 
# pass image into pytesseract module
# pytesseract is trained in many languages
image_to_text = pytesseract.image_to_string(dst, lang='eng')
 
# Print the text
print(image_to_text)

plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()