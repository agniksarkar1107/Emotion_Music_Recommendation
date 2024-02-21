import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser


from PIL import Image

im = Image.open('d990b9bd9184df62c7f86937ccc977f9.jpg')
st.set_page_config( page_title="Emovibe", page_icon = im ,layout="wide")  


new_image = im.resize((150, 100))
   


hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)  
  
       

model  = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("EMOVIBE")
st.image(new_image, caption='Emotion based song recommender')


st.write("Introducing EmoVibe, your personalized emotion-based song recommender webapp! Whether you're feeling joyful, contemplative, heartbroken, or pumped up, EmoVibe has got you covered. Simply select the emotion you're experiencing, and let our sophisticated algorithms curate a tailor-made playlist just for you. Say goodbye to endless scrolling and generic recommendations – EmoVibe understands your mood like no other. Discover new tunes that resonate with your feelings and enhance every moment. Let the music match your mood with EmoVibe.")
st.write("This webapp is maintained by Agnik Sarkar")

if "run" not in st.session_state:
	st.session_state["run"] = "true"

try:
	emotion = np.load("emotion.npy")[0]
except:
	emotion=""

if not(emotion):
	st.session_state["run"] = "true"
else:
	st.session_state["run"] = "false"

class EmotionProcessor:
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")

		##############################
		frm = cv2.flip(frm, 1)

		res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

		lst = []

		if res.face_landmarks:
			for i in res.face_landmarks.landmark:
				lst.append(i.x - res.face_landmarks.landmark[1].x)
				lst.append(i.y - res.face_landmarks.landmark[1].y)

			if res.left_hand_landmarks:
				for i in res.left_hand_landmarks.landmark:
					lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
					lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
			else:
				for i in range(42):
					lst.append(0.0)

			if res.right_hand_landmarks:
				for i in res.right_hand_landmarks.landmark:
					lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
					lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
			else:
				for i in range(42):
					lst.append(0.0)

			lst = np.array(lst).reshape(1,-1)

			pred = label[np.argmax(model.predict(lst))]

			print(pred)
			cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

			np.save("emotion.npy", np.array([pred]))

			
		drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
								landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
								connection_drawing_spec=drawing.DrawingSpec(thickness=1))
		drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
		drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)


		##############################

		return av.VideoFrame.from_ndarray(frm, format="bgr24")

lang_out = st.text_input("Language")
singer_out = st.text_input("Singer")

if lang_out and singer_out and st.session_state["run"] != "false":
	webrtc_streamer(key="key", desired_playing_state=True,
				video_processor_factory=EmotionProcessor)

btn = st.button("Recommend me songs")

if btn:
	if not(emotion):
		st.warning("Please let me capture your emotion first")
		st.session_state["run"] = "true"
	else:
		webbrowser.open(f"https://www.youtube.com/results?search_query={lang_out}+{emotion}+song+{singer_out}")
		np.save("emotion.npy", np.array([""]))
		st.session_state["run"] = "false"
