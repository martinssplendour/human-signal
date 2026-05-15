import base64

import cv2
import numpy as np


def decode_data_url_image(image: str):
    raw = image.split(",", 1)[1] if "," in image else image
    data = base64.b64decode(raw)
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode frame")
    return frame
