import streamlit as st
import numpy as np
import os
import cv2
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# INITIALIZE
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# Content for the sidebar
sidebar_content = """
    ## Registered Faces
    - Jenna Ortega
    - Robert Downey
    - Sardor Abdir
    - Taylor Swift
    - Alfajrine
"""

#RTC_CONFIGURATION
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": 'turns:freestun.tel:5350'}]}
)

recognized_names = []  # List to store recognized names

class VideoProcessor(VideoTransformerBase):
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        # frame.flags.writeable = False
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
        
        confidence_threshold = 0.5

        for x, y, w, h in faces:
            #cropped_img = rgb_img[y:y+h, x:x+w]
            img = rgb_img[y:y+h, x:x+w]
            img = cv2.resize(img, (160, 160))
            img = np.expand_dims(img, axis=0)
            ypred = facenet.embeddings(img)
            face_name = model.predict(ypred)
            conf = model.predict_proba(ypred)

            if conf.max() < confidence_threshold:
                final_name = 'unknown'
            else:
                final_name = encoder.inverse_transform(face_name)[0]
        
            #recognized_names.append(final_name)  # Store recognized names

            cv2.rectangle(image, (x, y), (x+w, y+h), (64, 224, 208), 10)
            cv2.putText(image, str(final_name), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 105, 180), 3, cv2.LINE_AA)
            
            return av.VideoFrame.from_ndarray(image, format="bgr24")


st.sidebar.markdown(sidebar_content)

st.write("**Face Recognition**")
    
webrtc_streamer(
    key="facerecognition",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)
