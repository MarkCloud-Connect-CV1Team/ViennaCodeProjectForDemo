#=======================================
# 모듈불러오기
#=======================================
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
from PIL import Image

#=======================================
# 모델로드 및 전처리
#=======================================
model = YOLO("./main/save_weight_class9/best.pt") 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#=======================================
# 함수정의
#=======================================

def get_object_description(object_code):
    object_dict = {
        '010102': '010102 : 하나의 별',
        '010103': '010103 : 두 개의 별',
        '010104': '010104 : 세 개의 별',
        '010105': '010105 : 네 개 이상의 별',
        '010109': '010109 : 사각별',
        '010110': '010110 : 오각 이상의 별',
        '010112': '010112 : 꼭지점이 불균등한 별',
        '010114': '010114 : 불완전한(미완성의) 별',
        '010115': '010115 : 혜성, 유성'
    }

    return object_dict.get(object_code, 'Unknown Object Code')

#=======================================
# Streamlit 앱
#=======================================
if __name__ == "__main__":
    # UI및 버튼
    st.title('Vienna Code Object Detection(YOLOv8)')
    st.markdown('Upload an image to detect objects.')
    uploaded_file = st.file_uploader("이미지를 업로드하세요.", type=["jpg", "jpeg", "png"])

    # 파일처리
    if uploaded_file is not None:
        # 업로드된 이미지를 읽어서 object detection 수행
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        results = model.predict(image,show_labels=True,iou=0.5,save=True,top_k=5)
        results[0].save_txt('./result.txt')	
        
        object_code = get_object_description(results[0].verbose().split()[1].split(",")[0])
        
        # 결과 이미지 보여주기
        img_with_boxes = results[0].plot()  # object detection 결과 이미지 생성
        img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
        st.image(img_with_boxes_rgb, caption=f'{object_code}', use_column_width=False, width=640)
        
        
        