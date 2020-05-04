import mss
import cv2
import numpy as np
import time


sct = mss.mss()
# Part of the screen to capture
tokens_rect = {"top": 450, "left": 700, "width": 550, "height": 160}
damage_rect = {"top": 720, "left": 810, "width": 130, "height": 70}
accuracy_rect = {"top": 720, "left": 960, "width": 130, "height": 55}

while "Screen capturing":
    last_time = time.time()

    # Get raw pixels from the screen, save it to a Numpy array
    img = np.array(sct.grab(accuracy_rect))
    img2 = np.array(sct.grab(damage_rect))
    img3 = np.array(sct.grab(tokens_rect))

    # Display the picture
    cv2.imshow("OpenCV/Numpy normal", img)
    cv2.imshow("OpenCV/Numpy normal2", img2)
    cv2.imshow("OpenCV/Numpy normal3", img3)


    print("fps: {}".format(1 / (time.time() - last_time)))

    # Press "q" to quit
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
