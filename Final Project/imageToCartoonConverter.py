import cv2

import imageCartoonizerKMeans
import imageCartoonizer
import numpy as np


def execute(frame):
    im = frame
    cv2.imwrite("capturedRawImage.jpg", im)
    # Crop the captured image
    image = frame[topLeftRow - 100:topLeftRow + 300, topLeftCol - 100:topLeftCol + 300]
    img1 = imageCartoonizer.cartoonize(image)
    img2 = imageCartoonizerKMeans.cartoonize(image)
    cv2.imwrite("imageCartoonizer-ConvertedImage.jpg", img1)
    cv2.imwrite("imageCartoonizerKMeans-ConvertedImage.jpg", img2)


if __name__ == "__main__":
    """ 
    run as >>python imageToCartoonConverter.py
    """
    captureImage = False
    try:
        cascadeClassifierPath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadeClassifierPath)

        video_capture = cv2.VideoCapture(0)

        while True:
            # Capture each frame
            ret, frame = video_capture.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Face detection from the captured frame
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            frameCopy = np.copy(frame)
            # Draw a rectangle around the faces for easier identification
            for (x, y, w, h) in faces:
                cv2.rectangle(frameCopy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                topLeftCol = x
                topLeftRow = y
                imRectWidth = w
                imRectHeight = h

            # Display the resulting frame to user
            cv2.imshow('Video', frameCopy)
            # press q to quit without converting the image
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("break")
                break
            # press c to quit and convert the image
            elif cv2.waitKey(1) & 0xFF == ord('c'):
                print("capture")
                captureImage = True
                break

        if captureImage:
            execute(frame)
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("KeyboardInterrupt"+KeyboardInterrupt)
    except Exception:
        print("Exception"+Exception)
