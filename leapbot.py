import streamlit as st
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import time
from PIL import Image
from datetime import date

st.set_page_config(
    page_title='L.E.A.P Bot',
    # layout="wide",
    initial_sidebar_state="expanded",
)


with open("intents.json") as file:
	data = json.load(file)


	words = []
	labels = []
	docs_x = []
	docs_y = []

	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			wrds = nltk.word_tokenize(pattern)
			words.extend(wrds)
			docs_x.append(wrds)
			docs_y.append(intent["tag"])

		if intent["tag"] not in labels:
			labels.append(intent["tag"])

	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
	words = sorted(list(set(words)))


	labels = sorted(labels)


	training = []
	output = []

	out_empty = [0 for _ in range(len(labels))]

	for x, doc in enumerate(docs_x):
		bag = []

		wrds = [stemmer.stem(w) for w in doc]

		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)
		output_row = out_empty[:]
		output_row[labels.index(docs_y[x])] = 1

		training.append(bag)
		output.append(output_row)

	training = numpy.array(training)
	output = numpy.array(output)

	with open('data.pickle','wb') as f:
		pickle.dump((words, labels, training, output), f)


tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
	model.load("model.tflearn")
except:
	model = tflearn.DNN(net)
	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
	model.save("model.tflearn")

def bag_of_words(s,words):
	bag = [0 for _ in range(len(words))]


	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1

	return numpy.array(bag)

from requests import Response


unique_key = 0
def chat():
	unique_key = 0
	q,x,p = st.columns([1,2,1])
	x.title("L.E.A.P Bot🤖")
	st.write('---')

	# Initialize chat history
	if "messages" not in st.session_state:
		st.session_state.messages = []

	tab1, tab2 = st.tabs(['About', 'ChatBot'])
	with tab2:
	# Accepting user input
		with st.form("chat_input", clear_on_submit=True):
			a, b = st.columns([4, 1])
			inp = a.text_input("User_Input: ", key=unique_key, placeholder='User prompt💭', label_visibility="collapsed")
			# if inp.lower() == "quit":
			if inp.strip():
				pass
			results = model.predict([bag_of_words(inp,words)])[0]
			results_index = numpy.argmax(results)
			tag = labels[results_index]
			dail = '+2349025629246'
			submitted = b.form_submit_button("Send", use_container_width=True)
			st.session_state.messages.append({"role": "user", "content": inp})

			
			if submitted:
				if results[results_index] > 0.5:
					for tg in data["intents"]:
						if tg['tag'] == tag:
							responses = tg['responses']
					with st.chat_message("assistant"):
						resp = random.choice(responses)
						st.markdown(resp + "▌")
						st.markdown("\n")
						st.session_state.messages.append({"role": "assistant", "content": resp})
				else:
					z = "Sorry you feel this way\nMind me recommending a place that can provide you with the care you nend?"
					st.write("Sorry you feel this way\nMind me recommending a place that can provide you with the care you nend?")
					st.session_state.messages.append({"role": "assistant", "content": z})
		st.write('---')
		st.write('For more information please contact:')
		st.markdown('TechandHi: admin@techandhi.com')
		st.markdown(dail)
		st.info('LEAP Bot is designed to offer empathetic support and a safe space for individuals in emotional distress. However, it is important to note that LEAP Bot is not a substitute for professional mental health advice or treatment. If you are facing severe emotional difficulties or mental health crises, please seek assistance from a qualified mental health professional or contact a crisis hotline immediately.', icon="ℹ️")
		st.write('---')
		st.subheader('Chat History')
		st.write('---')
		# Display chat messages from history on app rerun
		for message in st.session_state.messages:
			with st.chat_message(message["role"]):
				st.markdown(message["content"])

	with tab1:
		st.subheader('About L.E.A.P Bot')
		st.write('---')

		st.write("<span style='font-size: 20px;'>Welcome to LEAP Bot, your empathetic companion dedicated to providing emotional support and a safe space for individuals in need. Our AI-powered assistant is here to listen, understand, and assist you during challenging times.<span>", unsafe_allow_html=True)
		st.write('---')
		st.write("<span style='font-size: 30px;'>Key Features:<span>", unsafe_allow_html=True)
		st.write("**Empathetic Listening:** LEAP Bot is designed to provide a non-judgmental space for you to express yourself and share your thoughts and feelings.")
		st.write("**Connecting You with Help:** If you're in immediate danger or need professional assistance, LEAP Bot can guide you to the right resources and encourage you to seek help.")
		st.write("**Motivation and Inspiration:** LEAP Bot offers words of encouragement and motivation to empower you in your journey towards a more positive outlook.")
		st.write('---')
		with st.expander('Depressive disorders prevalence'):
			image = Image.open('depressive.png')
			image2 = Image.open('depressive-1.png')
			st.image(image, caption='Depressive disorders prevalence MAP')
			st.image(image2, caption='Depressive disorders prevalence CHART')
		st.write('---')
		st.write("<span style='font-size: 20px;'>Why Choose LEAP Bot?<span>", unsafe_allow_html=True)
		st.write("**A Safe Space:** LEAP Bot is dedicated to creating a secure and non-judgmental environment where you can freely express yourself and seek support.")
		st.write("**24/7 Availability:** LEAP Bot is here for you around the clock, ensuring that you're never alone in your struggles.")
		st.write("**Continuous Improvement:** We are committed to regularly updating our resources and providing you with the most relevant and reliable support.")
		st.write("**Empowering Resilience:** With LEAP Bot, you'll find the strength and motivation to overcome challenges and move forward with confidence.")





# if __name__ == '__main__':
# 	chat()

chat()	

