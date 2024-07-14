import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

st.set_page_config(page_title='Predictions') 
st.subheader("Real Time Attendance System")

# Retrive The Data From the Database
with st.spinner("Retrieving Data from Redis..."):
    redis_face_db = face_rec.retrive_data(name='academic:register')
    st.dataframe(redis_face_db)
st.success("Data Succesfully Retrieved from Redis")
# Time
waitTime = 30 # time in sec
setTime = time.time()
realtimepred = face_rec.RealTimePred()

# Real-Time Prediction
# Streamlit Webrtc
# Callback Function
def video_frame_callback(frame):
    global setTime
    img = frame.to_ndarray(format="bgr24") # 3D numpy Array
    # Operation You can perfrom on the array
    pred_img = realtimepred.face_recognition(img,redis_face_db,'embeddings',thresh=0.5)
    
    timenow = time.time()
    difftime = timenow - setTime
    if difftime >= waitTime:
        # if i comment below line out, video runs fine after 30 seconds
        # if its present video stops and there is no update in redis database also
        realtimepred.save_logs_redis() 
        setTime = time.time()
        print('Save Data to Redis Database')
    
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")
webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback,
                rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
