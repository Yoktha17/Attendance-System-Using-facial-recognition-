import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av
from Home import face_rec
st.set_page_config(page_title='Registration Form') 

st.subheader("Registration Form")

# init Registration form
registration_form = face_rec.RegistrationForm()

# Step 1: Collect Person Name & Role
# Form
person_name = st.text_input(label='Name',placeholder='First & Last Name')
role = st.selectbox(label='Select Your Role',options=('Student','Teacher'))
# Step 2: Collecting Facial Embeddings of the person
def video_callback_func(frame):
    img = frame.to_ndarray(format='bgr24') #3D array bgr
    reg_img, embeddings = registration_form.get_embedding(img)
    # Two Step process
    # 1st Step: Save Data into local Computer txt
    # ab = append in bytes
    if embeddings is not None:
        with open('face_embedding.txt',mode='ab') as f:
            np.savetxt(f,embeddings)
    
    return av.VideoFrame.from_ndarray(reg_img,format='bgr24')
webrtc_streamer(key='registration',video_frame_callback=video_callback_func,
                rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
# Step 3: Save Data into Redis Database


if st.button('submit'):
    return_val = registration_form.save_data_in_redis_db(person_name,role)
    if return_val == True:
        st.success(f"{person_name} Registered Successfully")
    elif return_val == 'name_false':
        st.error('Either name is empty or contains spaces')
    elif return_val == 'file_false':
        st.error(f"File_embedding Error!")