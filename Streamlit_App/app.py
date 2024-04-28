import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from time import sleep
import pandas as pd

# Load pre-trained emotion detection model
classifier = load_model('model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to detect emotions from the webcam feed
def detect_emotions(max_iterations=40):
    song_data = pd.read_csv('songs_data.csv')  # Replace 'song_data.csv' with your file path

    cap = cv2.VideoCapture(0)
    stframe = st.image([])
    count = 0
    detected_emotions=[]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        labels = []
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)
        count += 1
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                print(prediction)
                print(prediction.argmax())
                print(emotion_labels[prediction.argmax()])
                label=emotion_labels[prediction.argmax()]
                detected_emotions.append(label.lower())
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                stframe.image(frame, channels="RGB")
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count>=20:
            break
    stframe.empty()
    cap.release()
    cv2.destroyAllWindows()
    st.success('Emotion detection complete. Recommending songs...')
    # st.button("Recommend Songs")
    max_emotion = max(detected_emotions, key=detected_emotions.count)
    st.write(f"Your Mood is : {max_emotion}")

    # Filter songs based on the matching emotion in the 'song_emotion' column
    recommended_songs = song_data[song_data['song_emotion'] == max_emotion]

    # Recommend at least 25 songs matching the detected emotion
    if len(recommended_songs) >= 25:
        st.write("Recommended Songs:")
        for idx, row in recommended_songs.head(25).iterrows():
            st.markdown(f"[{row['name']}](https://open.spotify.com/track/{row['id']})")
    else:
        st.warning("Insufficient songs available for recommendation.")
    

# Streamlit UI
st.title('Emotion Detection and Song Recommendation App')

# Button to start emotion detection
if st.button('Start Emotion Detection'):
    detect_emotions(40)
