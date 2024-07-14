import numpy as np
import pandas as pd
import cv2
import redis
import os
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
# Time
import time
from datetime import datetime

# Connect to Redis Database
hostname = 'redis-10914.c326.us-east-1-3.ec2.redns.redis-cloud.com'
port = 10914
password = "LZecMR4plTbYvT6h6HsnXkkevhBG5stj"
r = redis.Redis(host=hostname, port=port,password=password)

# Retrive Data from Database
def retrive_data(name):
    retrive_dict = r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(lambda x:np.frombuffer(x,dtype=np.float32))
    index = retrive_series.index
    index = list(map(lambda x: x.decode(),index))
    retrive_series.index = index
    df = retrive_series.to_frame().reset_index()
    df.columns = ['name_role','embeddings']
    df[['Name','Role']] = df['name_role'].apply(lambda x:x.split('@')).apply(pd.Series)
    return df[['Name','Role','embeddings']]

# Configure Face Analysis
faceapp = FaceAnalysis(name='buffalo_l',
                     root='insightface_models',
                     providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0,det_size=(640,640),det_thresh=0.5)

# ML Search Algorithm
def ml_search_algorithm(dataframe, feature_column, test_vector, thresh=0.5):
    # Step 1: Take the dataframe which has the collection of data
    dataframe = dataframe.copy()
    # Step 2: Index Face Embedding from the dataframe and covert into array
    X_list = dataframe[feature_column].tolist()
    X = np.asarray(X_list)
    # Step 3: Calculate the Cosine Similarity
    similar = pairwise.cosine_similarity(X, test_vector.reshape(1, 512))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr
    # Step 4: Filter the data
    datafilter = dataframe.query(f'cosine >{thresh}')
    if len(datafilter) > 0:
        # Step 5: Get the Person Name
        datafilter.reset_index(drop=True, inplace=True)
        argmax = datafilter['cosine'].argmax()
        Name, Roll = datafilter.loc[argmax][['Name', 'Role']]
        # print(Name, Roll)
    else:
        Name = 'Unknown'
        Roll = 'Unknown'

    return Name, Roll

### Real Time Prediction
# Save logs for every 1 minute
class RealTimePred:
    def __init__(self):
        self.logs = dict(Name=[],Role=[],Current_time=[])
    def reset_dict(self):
        self.logs = dict(Name=[],Role=[],Current_time=[])
    def save_logs_redis(self):
        # Step 1: Create Logs Database
        dataframe = pd.DataFrame(self.logs)
        # Step 2: Drop the Duplicate Information
        dataframe.drop_duplicates('Name',inplace=True)
        # Step 3: Push Data to Redis Database
        # Encode the data
        name_list = dataframe['Name'].tolist()     
        role_list = dataframe['Role'].tolist()     
        ctime_list = dataframe['Current_time'].tolist() 
        encoded_data = []
        for name,role,time in zip(name_list,role_list,ctime_list):
            if name!='Unknown':
                concat_string = f"{name}@{role}@{time}"   
                encoded_data.append(concat_string)
        if len(encoded_data)>0:
            r.lpush('attendance:logs',*encoded_data)
        self.reset_dict()
        
    def face_recognition(self,test_image,dataframe, feature_column, thresh=0.5):
        # Step 0: Calculate Time
        curr_time = str(datetime.now())
        # Step 1: Take the test image and apply insightface
        test_result = faceapp.get(test_image)
        test_copy = test_image.copy()
        # Step 2: Use for loop and extract each embedding and pass to ml_search_algo
        for res in test_result:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = ml_search_algorithm(dataframe, feature_column, embeddings, thresh=thresh)
            if person_name == 'Unknown':
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(test_copy, (x1, y1), (x2, y2), color)
            cv2.putText(test_copy, person_name, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
            cv2.putText(test_copy, curr_time, (x1, y2+10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
            # Save Info in Logs Dict
            self.logs['Name'].append(person_name)
            self.logs['Role'].append(person_role)
            self.logs['Current_time'].append(curr_time)

        return test_copy

### Registration Form
class RegistrationForm:
    def __init__(self):
        self.sample=0
    def reset(self):
        self.sample=0
    
    def get_embedding(self,frame):
        # Get results from InsightFace Model
        results = faceapp.get(frame,max_num=1)
        embeddings = None
        for res in results:
            self.sample += 1
            x1,y1,x2,y2 = res['bbox'].astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0))
            # put text samples info
            text = f"samples = {self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)
            embeddings = res['embedding']
            
        return frame,embeddings
    def save_data_in_redis_db(self,name,role):
        # Name validation
        if name is not None:
            if name.strip()!='':
                key = f'{name}@{role}'
            else:
                return 'name_false'
        else:
            return 'file_false'
        # file Validation
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'    
        # Step 1: Load "face_embedding.txt"
        x_array = np.loadtxt('face_embedding.txt',dtype=np.float32) # flatten array
        # Step 2: Convert into array
        received_samples = int(x_array.size/512)
        x_array = x_array.reshape(received_samples,512)
        x_array = np.asarray(x_array)
        # Step 3: Cal. mean embeddings
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()
        # Step 4: Save data into redis
        r.hset(name='academic:register',key=key,value=x_mean_bytes)
        os.remove('face_embedding.txt')
        self.reset()
        
        return True