import streamlit as st
import face_rec
st.set_page_config(page_title='Attendance System',layout='wide') 
st.header("Attendance System Using Facial Recognition")

with st.spinner("Loading Models & Connecting to Redis..."):
    import face_rec
st.success("Models Loaded Succesfully")
st.success("Redis DB Successfully Connected")