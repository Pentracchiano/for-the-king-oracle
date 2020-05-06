import mss
import cv2
import numpy as np
import time
import pytesseract


sct = mss.mss()
# Part of the screen to capture: try numpy slicing to improve performance
tokens_rect = {"top": 450, "left": 700, "width": 550, "height": 160}
damage_rect = {"top": 720, "left": 810, "width": 130, "height": 70}
accuracy_rect = {"top": 725, "left": 960, "width": 130, "height": 45}

while "Screen capturing":
    last_time = time.time()

    # Get raw pixels from the screen, save it to a Numpy array
    accuracy_image = np.array(sct.grab(accuracy_rect))
    # damage_image = np.array(sct.grab(damage_rect))
    # tokens_image = np.array(sct.grab(tokens_rect))

    # preprocess accuracy_image
    accuracy_image = cv2.cvtColor(accuracy_image, cv2.COLOR_BGR2GRAY)
    accuracy_image = cv2.bitwise_not(accuracy_image)
    accuracy_image = cv2.GaussianBlur(accuracy_image, (3, 3), 0)
    _, accuracy_image = cv2.threshold(accuracy_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Display the picture
    cv2.imshow("OpenCV/Numpy normal", accuracy_image)
    # cv2.imshow("OpenCV/Numpy normal2", damage_image)
    # cv2.imshow("OpenCV/Numpy normal3", tokens_image)

    print(pytesseract.image_to_string(accuracy_image, config="--psm 6"))
    # print(pytesseract.image_to_string(damage_image))
    # print(pytesseract.image_to_string(tokens_image))

    print("fps: {}".format(1 / (time.time() - last_time)))

    # Press "q" to quit
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
