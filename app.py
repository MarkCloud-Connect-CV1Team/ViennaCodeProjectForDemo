# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="비엔나 코드 인식 서비스",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("비엔나 코드 인식 서비스")

# Sidebar
st.sidebar.header("Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 1, 100, 10)) / 100

# Selecting Detection
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Default Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    classes = {
                        0.0: '010102',
                        1.0: '010103',
                        2.0: '010104',
                        3.0: '010105',
                        4.0: '010109',
                        5.0: '010110',
                        6.0: '010112',
                        7.0: '010114',
                        8.0: '010115'
                    }
                    explanations = {
                        0.0: '하나의 별',
                        1.0: '두 개의 별',
                        2.0: '세 개의 별',
                        3.0: '네 개 이상의 별 [Note] 중분류 (01-11)의 성좌 및 성군은 포함하지 않음',
                        4.0: '사각별',
                        5.0: '오각 이상의 별',
                        6.0: '꼭지점이 불균등한 별 [Note] 꼭지점이 불균등한 별로 구성된 스파크를 포함',
                        7.0: '불완전한(미완성의) 별',
                        8.0: '혜성, 유성'
                        }
                    with st.expander("Detection Results"):
                        unique_class_index = set()

                        for r in res:
                            for i in range(len(r.boxes.cls)):
                                class_index = r.boxes.cls[i].item()

                                # 중복을 피하기 위해 이미 출력한 class_index면 무시
                                if class_index in unique_class_index:
                                    continue

                                unique_class_index.add(class_index)

                                # Map class index to class code using the classes dictionary
                                if class_index in classes:
                                    class_code = classes[class_index]
                                    explanation = explanations[class_index]
                                    st.write(f"{class_code} : {explanation}")
                                else:
                                    st.write(f"Unknown class index: {class_index}")
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")


else:
    st.error("Please select a valid source type!")
