﻿#######################################################
#  Initialise NLTK Inference
#######################################################
import re
import time
import tkinter as tk
from tkinter import filedialog
import aiml
import gensim
import numpy as np
import pandas
import requests
# from gensim.parsing.preprocessing import preprocess_documents
from nltk.inference import ResolutionProver
from nltk.sem import Expression
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import speech_recognition as sr
import pyttsx3

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Nutritionix API credentials
app_id = "456feebd"
app_key = "06ea5ed212cf32eeea201c7a617db6a2"

FOOD_CATEGORY = {
    0: ['burger', 'Burger'], 1: ['butter_naan', 'Butter Naan'], 2: ['chai', 'Chai'],
    3: ['chapati', 'Chapati'], 4: ['chole_bhature', 'Chole Bhature'], 5: ['dal_makhani', 'Dal Makhani'],
    6: ['dhokla', 'Dhokla'], 7: ['fried_rice', 'Fried Rice'], 8: ['idli', 'Idli'], 9: ['jalegi', 'Jalebi'],
    10: ['kathi_rolls', 'Kaathi Rolls'], 11: ['kadai_paneer', 'Kadai Paneer'], 12: ['kulfi', 'Kulfi'],
    13: ['masala_dosa', 'Masala Dosa'], 14: ['momos', 'Momos'], 15: ['paani_puri', 'Paani Puri'],
    16: ['pakode', 'Pakode'], 17: ['pav_bhaji', 'Pav Bhaji'], 18: ['pizza', 'Pizza'], 19: ['samosa', 'Samosa']
}

DRUG_CATEGORY = {
    0: 'Alaxan', 1: 'Bactidol', 2: 'Bioflu', 3: 'Biogesic', 4: 'DayZinc', 5: 'Decolgen',
    6: 'Fish Oil', 7: 'Kremil S', 8: 'Medicol', 9: 'Neozep'
}


def is_kb_consistent() -> bool:
    inconsistent = ResolutionProver().prove(None, kb)
    return not inconsistent


# Function to preprocess the questions by lemmatizing the words
def preprocess_documents(documents):
    lemmatizer = WordNetLemmatizer()
    preprocessed = []
    for document in documents:
        tokens = word_tokenize(document)
        lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
        preprocessed.append(lemmatized)
    return preprocessed


use_voice = False


def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        print("done")
    try:
        print("Yes")
        inputv = r.recognize_google(audio, show_all=False)
        print("check1")
        # print("You said: " + inputv)
        return inputv
    except sr.UnknownValueError:
        pass
    except sr.RequestError as e:
        pass
    return None


def get_answer(question):
    query_processed = preprocess_documents([question])[0]
    query_bow = questions_dict.doc2bow(query_processed)
    query_tfidf = tfidf[query_bow]
    similarities = index[query_tfidf]
    max_index = similarities.argmax()
    return answers_corpus[max_index]


def predict_image_food(filename, model) -> str:
    img_ = image.load_img(filename, target_size=(299, 299))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    prediction = model.predict(img_processed)

    index = np.argmax(prediction)
    return FOOD_CATEGORY[index][1]


def predict_image_drug(filename, model) -> str:
    img = image.load_img(filename, target_size=(300, 300))
    img = image.img_to_array(img, dtype=np.uint8)
    img = np.array(img) / 255.0
    p = model.predict(img[np.newaxis, ...])

    return DRUG_CATEGORY[np.argmax(p[0], axis=-1)]


# Function to retrieve information about a food item
def get_nutrition_info(food):
    url = f"https://api.nutritionix.com/v1_1/search/{food}?results=0:1&fields=item_name,brand_name,nf_calories,nf_total_fat"
    headers = {
        "x-app-id": app_id,
        "x-app-key": app_key
    }
    response = requests.get(url, headers=headers)
    return response.json()


#######################################################
#  Init vars
#######################################################
time.clock = time.time
read_expr = Expression.fromstring
root = tk.Tk()
root.withdraw()
model_food = load_model("model_food.h5")
model_drugs = load_model("model_drugs.h5")

#######################################################
#  Initialise Knowledgebase.
#######################################################
kb: list[Expression] = []
data = pandas.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]

if not is_kb_consistent():
    print('KB is inconsistent!')
    exit(1)

#######################################################
#  Initialise AIML agent
#######################################################
# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-logic.xml")

#######################################################
#  Initialise Q/A pairs
#######################################################
qa_data = pandas.read_csv('docs/Task A QA.csv', header=None)
questions_corpus = qa_data[0]
answers_corpus = qa_data[1]
questions_processed = preprocess_documents(questions_corpus)
questions_dict = gensim.corpora.Dictionary(questions_processed)
questions_bow = [questions_dict.doc2bow(text) for text in questions_processed]

tfidf = gensim.models.TfidfModel(questions_bow, smartirs='npu')
index = gensim.similarities.MatrixSimilarity(tfidf[questions_bow])
#######################################################
# Welcome user
#######################################################
print("Hello there! My name is Nutrino")
print("I'm here to assist you with any questions or concerns you have regarding health, wellness, and nutrition. Go ahead and ask me anything!")
print("If you would me to speak out my replies, please enter SPEAK to start and STOP to stop")
print("If you would like to ask a question using voice, please enter VOICE.")

use_voice = False
r = sr.Recognizer()

#######################################################
# Main loop
#######################################################
engine = pyttsx3.init()
# voice properties
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id) # select a male voice
engine.setProperty('rate', 160) # adjust speaking rate
engine.setProperty('volume', 0.9) # adjust volume
engine.setProperty('pitch', 1.1) # adjust pitch

while True:
    # get user input
    try:
        if not use_voice:
            user_input = input("> ")
        else:
            inputv = listen()
            if inputv is not None:  # Add this check
                print(f"You said: {inputv}")
                user_input = inputv
                use_voice = False
            else:
                continue

        if user_input.lower() == "voice":
            use_voice = True
            reply = "Voice Input Activated"
            print(reply)
            if speakout:
                engine.say(reply)
                engine.runAndWait()
            continue

        if user_input.lower() == "speak":
            print("Speech output activated")
            engine.say("Speech output activated")
            engine.runAndWait()
            speakout = True
            continue

        if user_input.lower() == "stop":
            print("Speech output deactivated")
            engine.say("Speech output deactivated")
            engine.runAndWait()
            speakout = False
            continue

    except (KeyboardInterrupt, EOFError):
        print("Bye!")
        break

    # pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    answer = ''

    # activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(user_input)

    if len(answer) == 0:
        continue

    # post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break

        # Here are the processing of the new logical component:
        elif cmd == 31:  # if input pattern is "I know that * is *"
            object, subject = params[1].split(' is ')
            expr = read_expr(subject + '(' + object + ')')

            kb.append(expr)
            if not is_kb_consistent():
                reply = "Sorry this contradicts with what I know!"
                print(reply)
                if speakout:
                    engine.say(reply)
                    engine.runAndWait()
                kb.pop(-1)
                continue
            reply = 'OK, I will remember that' + object + 'is' + subject
            print(reply)
            if speakout:
                engine.say(reply)
                engine.runAndWait()

        elif cmd == 32: # if the input pattern is "check that * is *"
            object, subject = params[1].split(' is ')
            expr = read_expr(subject + '(' + object + ')')
            answer = ResolutionProver().prove(expr, kb)
            if answer:
                reply = "That is correct"
                print(reply)
                if speakout:
                    engine.say(reply)
                    engine.runAndWait()

            else:
                reply = "Im not sure, let me check"
                print(reply)
                if speakout:
                    engine.say(reply)
                    engine.runAndWait()
                kb.append(expr)
                if is_kb_consistent():
                    reply = "Sorry I don't know"
                    print(reply)
                    if speakout:
                        engine.say(reply)
                        engine.runAndWait()
                else:
                    reply = "That is incorrect"
                    print(reply)
                    if speakout:
                        engine.say("Incorrect")
                        engine.runAndWait()
                kb.pop(-1)

        elif cmd == 33:
            for fact in kb:
                print(fact)
                if speakout:
                    engine.say(fact)
                    engine.runAndWait()

        elif cmd == 34:
            file_path = filedialog.askopenfilename()
            pred = predict_image_food(file_path, model_food)
            print(f"This could be a {pred}")
            if speakout:
                engine.say(f"This could be a {pred}")
                engine.runAndWait()

            nutrition_info = get_nutrition_info(pred)
            if 'hits' in nutrition_info:
                calories = nutrition_info['hits'][0]['fields']['nf_calories']
                protein = nutrition_info['hits'][0]['fields']['nf_protein']
                carbs = nutrition_info['hits'][0]['fields']['nf_total_carbohydrate']
                fat = nutrition_info['hits'][0]['fields']['nf_total_fat']
                if speakout:
                    engine.say("Speech output deactivated")
                    engine.runAndWait()
                print(f"Calories in 100g of {pred}: {calories}")
                print(f"Protein in 100g of {pred}: {protein}")
                print(f"Carbs in 100g of {pred}: {carbs}")
                print(f"Fat in 100g of {pred}: {fat}")
            else:
                print(f"No information found for {pred}")

        elif cmd == 35:
            file_path = filedialog.askopenfilename()
            pred = predict_image_drug(file_path, model_drugs)
            print(f"This could be a {pred}")
            if speakout:
                engine.say(f"This could be a {pred}")
                engine.runAndWait()

        elif cmd == 99:
            qa_answer = get_answer(user_input)
            print(qa_answer)
            if speakout:
                engine.say(qa_answer)
                engine.runAndWait()

    else:
        print(answer)
        if speakout:
            engine.say(answer)
            engine.runAndWait()