import streamlit as st
import time
import uuid 
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image 
import pandas as pd
import sqlite3
from rembg import remove
import datetime
import pickle
import streamlit.components.v1 as components
import base64
import warnings
warnings.filterwarnings("ignore")
import librosa
import librosa.display
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av






with st.sidebar:
    st.title("Music Recommender")
    choice=st.radio("Navigation",["Yeni şarkı önerin","Duyguya göre(Yüz Fotoğrafı)","Duyguya Göre(Ses Kaydı)"])
if choice=="Duyguya göre(Yüz Fotoğrafı)":
    music_df=pd.read_csv("music.csv")
    st.title("Reccomending Music Type :musical_score:")
    img=st.camera_input("Camera")
    model=load_model("my_face_model.h5")
    conn=sqlite3.connect("database.db")
    cursor=conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS emotions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emotion TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
    ''')






    def process_image(input_img):
        if input_img.mode == 'RGBA':
           input_img = input_img.convert('RGB')
        input_img=input_img.resize((170,170)) 
        input_img=np.array(input_img)
        input_img=input_img/255.0
        input_img=np.expand_dims(input_img,axis=0)
        return input_img
    class_names=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
    if img is not None:
        img=Image.open(img)
        img=remove(img)
        st.image(img,caption="Uploaded Image")
        image=process_image(img)
        prediction=model.predict(image)
        predicted_class=np.argmax(prediction)  
        st.write(class_names[predicted_class])
        emotion=class_names[predicted_class]
        emotions_df=music_df[music_df["emotion"]==emotion]
        reccomends=emotions_df["Title"].sample(5)
        st.write(reccomends)
        cursor.execute("INSERT INTO emotions (emotion) VALUES (?)", (emotion,))
        conn.commit()
        conn.close()









    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    one_week_ago = datetime.datetime.now() - datetime.timedelta(days=7)
    query = '''
SELECT emotion, COUNT(*) AS count
FROM emotions
WHERE timestamp >= ?
GROUP BY emotion
'''
    cursor.execute(query, (one_week_ago,))
    results = cursor.fetchall()
    emotion_counts = {row[0]: row[1] for row in results}
    angry_counts=emotion_counts.get("Angry",0)
    happy_counts=emotion_counts.get("Happy",0)
    surpise_counts=emotion_counts.get("Surprise",0)
    neutral_counts=emotion_counts.get("Neutral",0)
    sad_count = emotion_counts.get("Sad", 0)
    fear_count = emotion_counts.get("Fear", 0)
    disgust_count = emotion_counts.get("Disgust", 0)
    total=angry_counts+happy_counts+sad_count+surpise_counts+neutral_counts+fear_count+fear_count
    if total > 20:
        if (sad_count + fear_count + disgust_count) > 20:
            st.write("Hayatta daha iyi olan şeylere odaklanarak daha mutlu ve sağlıklı bir yaşam sürebilirsin.")
        elif (sad_count + fear_count + disgust_count) < 3:
            st.write("Hayatta mutlu olmak gibisi yok.") 
        elif (sad_count + fear_count + disgust_count) < 10:
            st.write("Hayatta küçük değişiklikler mutluluğu sana getirebilir.")
    conn.close()







if choice=="Yeni şarkı önerin":
    st.title("Eğer önerileri beğenmediyseniz yeni bir şarkı önerebilirsiniz.")
    st.write("Aşağıdakilerden her biri için bir değer girmeniz gerekiyor.")
    file_path = 'music_model.pkl'
    with open(file_path, 'rb') as file:
        loaded_model = pickle.load(file)
    name = st.text_input("Şarkının adı")
    artist = st.text_input("Sanatçı")
    m_type = st.text_input("Şarkı tipi")
    bpm = st.number_input("Dakika Başına Vuruş (BPM)", 37, 206)
    energy = st.number_input("Enerji", 3, 100)
    dance = st.number_input("Dans Edilebilirlik", 10, 96)
    volume = st.number_input("Ses Yüksekliği", -27, -2)
    live = st.number_input("Canlılık", 2, 99)
    ac = st.number_input("Akustiklik", 0, 99)
    speech = st.number_input("Konuşkanlık", 2, 55)
    popularity = st.number_input("Popülerlik", 11, 100)
    is_data_complete = all([
    name,
    artist,
    m_type,
    bpm,
    energy,
    dance,
    volume,
    live,
    ac,
    speech,
    popularity,
              ])
    if st.button("Ekleyin"):
       if not is_data_complete:
          st.warning("Lütfen tüm alanları doldurun.")
       else:   
          energy*=2
          dance*=2
          volume*=-1
          popularity/=2
          speech*=2
        
          inputs=[[bpm,energy,dance,volume,live,ac,speech,popularity]]
          d={0:"Neutral",1:"Angry",2:"Sad",3:"Happy",4:"Fear",5:"Disgust",6:"Surprise"}
          prediction = loaded_model.predict(inputs)
          key = prediction[0]
          predicted_emotion = d.get(key, "Bulunamadı, Lütfen Tekrar deneyin.")
          st.write(predicted_emotion)
          music_df=pd.read_csv("music.csv")
          new_index=len(music_df)+1
          new_row={
              "Index":new_index,
              "Title":name,
              "Artist":artist,
              "Top Genre":m_type,
              "Year":0,
              "Beats Per Minute (BPM)":bpm,
              "Energy":energy,
              "Danceability":dance,
              "Loudness (dB)":volume,
              "Liveness":live,
              "Valence":0,
              "Length (Duration)":0,
              "Acousticness":ac,
              "Speechiness":speech,
              "Popularity":popularity,
              "emotion":predicted_emotion,
              "cluster":prediction
          }
          music_df = pd.concat([music_df, pd.DataFrame([new_row])], ignore_index=True)
          music_df.to_csv("music.csv",index=False)















if choice=="Duyguya Göre(Ses Kaydı)":
    voice_model=load_model("voice_model.h5")
    st.title("Ses Dosyası Yükleyin")
    class AudioRecorder(AudioProcessorBase):
        def recv_audio(self, frames: av.AudioFrame) -> av.AudioFrame:
            return frames
    st.write("Ses kaydı almak için aşağıdaki butonu kullanın:")
    webrtc_ctx = webrtc_streamer(
    key="audio_recorder",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioRecorder, 
    media_stream_constraints={"audio": True, "video": False},)
    if webrtc_ctx.audio_receiver:
       audio_frames = list(webrtc_ctx.audio_receiver.get_frames())
       if audio_frames:
          audio_np = np.concatenate([frame.to_ndarray() for frame in audio_frames])
          n_mels = 128
          mel_spectrogram = librosa.feature.melspectrogram(audio_np, sr=None, n_mels=n_mels, fmax=8000)
          prediction=voice_model.predict(mel_spectrogram)   
          class_names=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
          emotion=class_names[np.argmax(prediction)]
          st.write(emotion)
          emotions_df=music_df[music_df["emotion"]==emotion]
          reccomends=emotions_df["Title"].sample(5)
          st.write(reccomends)
          cursor.execute("INSERT INTO emotions (emotion) VALUES (?)", (emotion,))
          conn.commit()
          conn.close()









    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS emotions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emotion TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
    ''')
    one_week_ago = datetime.datetime.now() - datetime.timedelta(weeks=1)
    query = '''
SELECT emotion, COUNT(*) AS count
FROM emotions
WHERE timestamp >= ?
GROUP BY emotion
'''
    cursor.execute(query, (one_week_ago,))
    results = cursor.fetchall()
    emotion_counts = {row[0]: row[1] for row in results}
    angry_counts=emotion_counts.get("Angry",0)
    happy_counts=emotion_counts.get("Happy",0)
    surpise_counts=emotion_counts.get("Surprise",0)
    neutral_counts=emotion_counts.get("Neutral",0)
    sad_count = emotion_counts.get("Sad", 0)
    fear_count = emotion_counts.get("Fear", 0)
    disgust_count = emotion_counts.get("Disgust", 0)
    total=angry_counts+happy_counts+sad_count+surpise_counts+neutral_counts+fear_count+fear_count
    if total > 20:
        if (sad_count + fear_count + disgust_count) > 20:
            st.write("Hayatta daha iyi olan şeylere odaklanarak daha mutlu ve sağlıklı bir yaşam sürebilirsin.")
        elif (sad_count + fear_count + disgust_count) < 3:
            st.write("Hayatta mutlu olmak gibisi yok.") 
        elif (sad_count + fear_count + disgust_count) < 10:
            st.write("Hayatta küçük değişiklikler mutluluğu sana getirebilir.")
    conn.close()