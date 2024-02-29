import cv2
import streamlit as st
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#INITIALIZE
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

def main():
    st.sidebar.markdown(sidebar_content)

    st.write("**Face Recognition**")
    frame_placeholder = st.empty()
    detected_faces_placeholder = st.empty()
    # cap_index = 0
    # cap = cv2.VideoCapture(cap_index)
    
    # # If unable to open, try with index 1
    # if not cap.isOpened():
    #     st.warning(f'Warning: Unable to open the camera at index {cap_index}. Trying with index 1.')
    #     cap_index = 1
    #     cap = cv2.VideoCapture(cap_index)

    #     # If still unable to open, display an error message and stop the app
    #     if not cap.isOpened():
    #         st.error(f'Error: Unable to open the camera at indices 0 and 1.')
    #         st.stop()
    # Check additional camera indices
    for i in range(2):  # Try indices from 0 to 1
        try:
            cap_additional = cv2.VideoCapture(i)
            if cap_additional.isOpened():
                st.write(f"Camera index {i} is available.")
                cap_additional.release()
            else:
                st.warning(f"Camera index {i} is not available.")
        except cv2.error as e_additional:
            st.error(f"Error opening camera index {i}: {e_additional}")
            st.warning(f"Please check camera permissions and make sure no other application is using camera index {i}.")

    confidence_threshold = 0.5
    cropped_imgs=[]
    frames_processed = 0
    update_interval = 20  # Update every 5 frames
    placeholder = st.empty()
    while cap.isOpened():
        _, frame = cap.read()
        #frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
        for x,y,w,h in faces:
            cropped_img = rgb_img[y:y+h, x:x+w]
            img = rgb_img[y:y+h, x:x+w]
            img = cv2.resize(img, (160,160)) # 1x160x160x3
            img = np.expand_dims(img,axis=0)
            ypred = facenet.embeddings(img)
            face_name = model.predict(ypred)
            conf = model.predict_proba(ypred)

            #save cropped face images
            if cropped_img is not None:
                cropped_imgs.append(cropped_img)
            if len(cropped_imgs)>6:
                cropped_imgs.pop(0)
            #cropped_images = np.array(cropped_imgs)

            if conf.max() < confidence_threshold:
                final_name = 'unknown'
            else:
                final_name = encoder.inverse_transform(face_name)[0]

            #final_name = encoder.inverse_transform(face_name)[0]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 10)
            cv2.putText(frame, str(final_name), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 3, cv2.LINE_AA)
            print(final_name)
            placeholder.write("Detected Faces")
            frame_placeholder.image(frame, use_column_width=True)
            
            
            col1, col2, col3, col4, col5 = st.columns(5)

            frames_processed += 1
            if frames_processed % update_interval == 0:
                # Clear the previous content and update detected faces
                detected_faces_placeholder.empty()
                with detected_faces_placeholder:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    for i, col in enumerate([col1, col2, col3, col4, col5]):
                        if i < len(cropped_imgs):
                            im = cv2.cvtColor(cropped_imgs[i], cv2.COLOR_BGR2RGB)
                            col.image(im, use_column_width=True)

    cap.release()



    st.success('Video is processed')
    st.stop()

if __name__ == '__main__':
    main()
