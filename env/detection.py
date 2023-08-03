from PIL import Image
import numpy as np
import cv2
import pytesseract
from deep_translator import GoogleTranslator, single_detection
import deep_translator.exceptions

# print(pytesseract.image_to_string(Image.open('./image-to-text-detection-python/images.png')))

img_cv = cv2.imread(r'./images.png')
# by default opencv stores images in BGR format and since pytesseract assumes
# RGB format, we need to convert from BGR to RGB format/mode
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
# print(pytesseract.image_to_string(img_rgb))

### Detecting characters 
wImg, hImg, _ = img_rgb.shape
boxes = pytesseract.image_to_boxes(img_rgb)
for b in boxes.splitlines():
    b = b.split(' ')
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    print(x,y,w,h)
    cv2.rectangle(img_rgb, (x, wImg-y), (w, wImg-h), (0,0,255), 2)
    cv2.putText(img_rgb, b[0], (x, wImg-y+15), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 2)

# ### Detecting words
wImg, hImg, _ = img_rgb.shape
boxes = pytesseract.image_to_data(img_rgb)
print(boxes)
for x, b in enumerate(boxes.splitlines()):
    if x != 0:
        b = b.split()
        if len(b) == 12:
            x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            print(x,y,w,h)
            cv2.rectangle(img_rgb, (x, y), (w+x, h+y), (0,0,255), 2)
            cv2.putText(img_rgb, b[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 2)

### Detecting digits
wImg, hImg, _ = img_rgb.shape
conf = r'--oem 3 --psm 6 outputbase digits'
boxes = pytesseract.image_to_data(img_rgb)
for x, b in enumerate(boxes.splitlines()):
    if x != 0:
        b = b.split()
        if len(b) == 12:
            x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            cv2.rectangle(img_rgb, (x, y), (w+x, h+y), (0,0,255), 2)
            text = b[11]
            try:
                lang = single_detection(text, api_key='849df3264c91bffcf6a76454a701426e')
                try:
                    translated_text = GoogleTranslator(source='auto', target='english').translate(text)
                    cv2.putText(img_rgb, translated_text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 2)
                except deep_translator.exceptions.InvalidSourceOrTargetLanguage:
                    cv2.putText(img_rgb, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 2)
            except IndexError:
                cv2.putText(img_rgb, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 2)

cv2.imshow('Result', img_rgb)
cv2.waitKey(0)
