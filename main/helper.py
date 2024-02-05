from ultralytics import YOLO
import time
import streamlit as st
import cv2
import PIL
import uuid
from datetime import datetime
import pytz
import settings


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (640, int(640*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)
        
        
        
# save result
def save_detection_results(result, uploaded_image):
    try:
        result.save_txt('./save/result.txt', save_conf=True)
        result.tojson('./save/result.json')	
        im_array = result.plot()
        im = PIL.Image.fromarray(im_array[:, :, ::-1])
        # saved_image_path = f'./save/{uuid.uuid4()}_{str(uploaded_image.name)}'
        # 현재 날짜와 시간을 이용하여 파일명 생성
        korea_timezone = pytz.timezone('Asia/Seoul')
        current_time = datetime.now(korea_timezone).strftime("%Y%m%d_%Hh%Mm%Ss")
        saved_image_path = f'./save/{current_time}_{str(uploaded_image.name)}'

        im.save(saved_image_path)
        st.success("Detection results saved successfully!")
    except Exception as ex:
        st.error("Error occurred while saving detection results.")
        st.error(ex)