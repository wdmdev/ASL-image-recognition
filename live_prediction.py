import numpy as np
import cv2
from prediction_chooser import predict_frame
from img_loader import preprocess_frame


def predict_live_video(model):
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        frame = frame[:200,:200]
        cv2.imshow('frame',frame)
        f = preprocess_frame(frame)
        print(predict_frame(model, f))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
