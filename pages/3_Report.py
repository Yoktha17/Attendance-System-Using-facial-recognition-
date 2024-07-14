import streamlit as st
from Home import face_rec
st.set_page_config(page_title='Reporting',layout='centered') 
st.subheader("Reporting")

# Retrieve Logs Data & And show in Report.py
# Extract data from redis list
name = 'attendance:logs'
def load_logs(name,end=-1):
    logs_list = face_rec.r.lrange(name,start=0,end=end)
    return logs_list

# Tabs:
tab1, tab2 = st.tabs(['Registered Data','Logs'])
with tab1:
    if st.button("Refresh Data"):
        with st.spinner("Retrieving Data from Redis..."):
            redis_face_db = face_rec.retrive_data(name='academic:register')
            st.dataframe(redis_face_db) 
with tab2:
    if st.button('Refresh Logs'):
        st.write(load_logs(name=name))