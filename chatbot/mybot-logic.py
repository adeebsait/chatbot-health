#######################################################
#  Initialise NLTK Inference
#######################################################
# from gensim.parsing.preprocessing import preprocess_documents
import csv
import json
import os
import random
import time
import tkinter as tk
from tkinter import filedialog

import aiml
import gensim
import numpy as np
import pandas
import pyttsx3
import requests
import speech_recognition as sr
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
from nltk.inference import ResolutionProver
from nltk.sem import Expression
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import torchvision.transforms as T
from yolov5 import YOLOv5

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

FOOD_CATEGORY = {
    0: ['burger', 'Burger'], 1: ['butter_naan', 'Butter Naan'], 2: ['chai', 'Chai'],
    3: ['chapati', 'Chapati'], 4: ['chole_bhature', 'Chole Bhature'], 5: ['dal_makhani', 'Dal Makhani'],
    6: ['dhokla', 'Dhokla'], 7: ['fried_rice', 'Fried Rice'], 8: ['idli', 'Idli'], 9: ['jalegi', 'Jalebi'],
    10: ['kathi_rolls', 'Kaathi Rolls'], 11: ['kadai_paneer', 'Kadai Paneer'], 12: ['kulfi', 'Kulfi'],
    13: ['masala_dosa', 'Masala Dosa'], 14: ['momos', 'Momos'], 15: ['paani_puri', 'Paani Puri'],
    16: ['pakode', 'Pakode'], 17: ['pav_bhaji', 'Pav Bhaji'], 18: ['pizza', 'Pizza'], 19: ['samosa', 'Samosa']
}

FOODS_CATEGORY = {
    0: 'hot-dog', 1: 'Apple', 2: 'Artichoke', 3: 'Asparagus', 4: 'Bagel', 5: 'Baked-goods',
    6: 'Banana', 7: 'Beer', 8: 'Bell-pepper', 9: 'Bread', 10: 'Broccoli', 11: 'Burrito',
    12: 'Cabbage', 13: 'Cake', 14: 'Candy', 15: 'Cantaloupe', 16: 'Carrot', 17: 'Common-fig',
    18: 'Cookie', 19: 'Dessert', 20: 'French-fries', 21: 'Grape', 22: 'Guacamole', 23: 'Hot-dog',
    24: 'Ice-cream', 25: 'Muffin', 26: 'Orange', 27: 'Pancake', 28: 'Pear', 29: 'Popcorn',
    30: 'Pretzel', 31: 'Strawberry', 32: 'Tomato', 33: 'Waffle', 34: 'food-drinks', 35: 'Cheese',
    36: 'Cocktail', 37: 'Coffee', 38: 'Cooking-spray', 39: 'Crab', 40: 'Croissant', 41: 'Cucumber',
    42: 'Doughnut', 43: 'Egg', 44: 'Fruit', 45: 'Grapefruit', 46: 'Hamburger', 47: 'Honeycomb',
    48: 'Juice', 49: 'Lemon', 50: 'Lobster', 51: 'Mango', 52: 'Milk', 53: 'Mushroom',
    54: 'Oyster', 55: 'Pasta', 56: 'Pastry', 57: 'Peach', 58: 'Pineapple', 59: 'Pizza',
    60: 'Pomegranate', 61: 'Potato', 62: 'Pumpkin', 63: 'Radish', 64: 'Salad', 65: 'food',
    66: 'Sandwich', 67: 'Shrimp', 68: 'Squash', 69: 'Squid', 70: 'Submarine-sandwich',
    71: 'Sushi', 72: 'Taco', 73: 'Tart', 74: 'Tea', 75: 'Vegetable', 76: 'Watermelon', 77: 'Wine',
    78: 'Winter-melon', 79: 'Zucchini', 80: 'Banh_mi', 81: 'Banh_trang_tron',
    82: 'Banh_xeo', 83: 'Bun_bo_Hue', 84: 'Bun_dau', 85: 'Com_tam', 86: 'Goi_cuon', 87: 'Pho',
    88: 'Hu_tieu', 89: 'Xoi'
}

DRUG_CATEGORY = {
    0: 'Alaxan', 1: 'Bactidol', 2: 'Bioflu', 3: 'Biogesic', 4: 'DayZinc', 5: 'Decolgen',
    6: 'Fish Oil', 7: 'Kremil S', 8: 'Medicol', 9: 'Neozep'
}

use_voice = False


def is_kb_consistent() -> bool:
    inconsistent = ResolutionProver().prove(None, kb)
    return not inconsistent


def print_kb():
    categories = {}

    for imp_expr in kb:
        line = str(imp_expr)
        parts = line.split(" -> ")
        category = parts[0].strip()

        if "(" in category:
            category = category.split("(")[0]

        if category not in categories:
            categories[category] = []

        if len(parts) > 1:
            content = parts[1].strip()
        else:
            content = ""

        if content.startswith("-"):
            content = "not " + content[1:]
        content = content.replace(" & ", " and ")
        content = content.replace(" -> ", ": ")

        if "(" in line:
            content = line.replace("(", " is ").replace(")", "")

        # Replace '-' symbol with 'not '
        content = content.replace("-", "not ")

        categories[category].append(content)

    for category, items in categories.items():
        print(f"{category}:")
        for item in items:
            print(f"  - {item}")
        print()


# Function to preprocess the questions by lemmatizing the words
def preprocess_documents(documents):
    lemmatiser = WordNetLemmatizer()
    preprocessed = []
    for document in documents:
        tokens = word_tokenize(document)
        lemmatised = [lemmatiser.lemmatize(word) for word in tokens]
        preprocessed.append(lemmatised)
    return preprocessed


def read_csv_file(file_name):
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        restaurant_data = {row["name"]: (int(row["service"]), int(row["food"])) for row in reader}
    return restaurant_data


def triangular_membership(x, a, b, c):
    epsilon = 1e-6  # small constant to avoid division by zero
    return np.fmax(np.fmin((x - a) / (b - a + epsilon), (c - x) / (c - b + epsilon)), 0)


def evaluate_restaurant(service, food):
    # Service quality fuzzy sets (low, medium, high)
    service_low = triangular_membership(service, 0, 0, 5)
    service_medium = triangular_membership(service, 0, 5, 10)
    service_high = triangular_membership(service, 5, 10, 10)

    # Food quality fuzzy sets (low, medium, high)
    food_low = triangular_membership(food, 0, 0, 5)
    food_medium = triangular_membership(food, 0, 5, 10)
    food_high = triangular_membership(food, 5, 10, 10)

    # Fuzzy rules
    rule1 = np.fmin(service_low, food_low)
    rule2 = np.fmin(service_medium, food_low)
    rule3 = np.fmin(service_high, food_low)
    rule4 = np.fmin(service_low, food_medium)
    rule5 = np.fmin(service_medium, food_medium)
    rule6 = np.fmin(service_high, food_medium)
    rule7 = np.fmin(service_low, food_high)
    rule8 = np.fmin(service_medium, food_high)
    rule9 = np.fmin(service_high, food_high)

    # Output fuzzy sets (low, medium, high)
    quality_low = np.fmax(rule1, rule4)
    quality_medium = np.fmax(np.fmax(rule2, rule5), rule7)
    quality_high = np.fmax(np.fmax(rule3, rule6), rule9)

    return quality_low, quality_medium, quality_high


def defuzzify(low_quality, medium_quality, high_quality):
    # Define output crisp values for low, medium, and high quality
    crisp_low = 0
    crisp_medium = 0.5
    crisp_high = 1

    # Calculate the centroid (weighted average)
    total_weight = low_quality + medium_quality + high_quality
    crisp_output = (
                           low_quality * crisp_low + medium_quality * crisp_medium + high_quality * crisp_high
                   ) / total_weight

    return crisp_output


def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        inputv = r.recognize_google(audio, show_all=False)
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


def get_exercises_for_body_part(body_part):
    exercise_data = pandas.read_csv('megaGymDataset.csv')
    exercises = exercise_data[exercise_data['BodyPart'].str.lower() == body_part.lower()]['Title'].tolist()
    random_exercises = random.sample(exercises, k=min(5, len(exercises)))
    return random_exercises


def predict_image_food(filename, model) -> str:
    img_ = image.load_img(filename, target_size=(299, 299))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    prediction = model.predict(img_processed)

    index = np.argmax(prediction)
    return FOOD_CATEGORY[index][1]


def detect_foods(file_path, food_model, class_names, conf_threshold=0.3):  # Add a confidence threshold parameter
    # Load image
    image = Image.open(file_path)

    # Prepare image for the model
    transform = T.Compose(
        [T.Resize((640, 640)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = transform(image).unsqueeze(0)

    # Get predictions from the model
    results = food_model.predict(image)

    # Access the detected objects
    detections = results[0].tolist()

    # Process the detections
    predicted_foods = []
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection[:6]
        if conf >= conf_threshold:  # Only include detections with confidence above the threshold
            class_name = class_names[int(class_id)]  # Use the class_names list
            predicted_foods.append(class_name)

    # Remove duplicate food names
    unique_foods = list(set(predicted_foods))

    return unique_foods


def predict_image_food_multi(filename, model, n=3) -> list:
    img_ = image.load_img(filename, target_size=(299, 299))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    prediction = model.predict(img_processed)

    top_n_indices = np.argsort(prediction[0])[-n:]
    top_n_predictions = [FOOD_CATEGORY[i][1] for i in top_n_indices]

    return top_n_predictions


def predict_image_drug(filename, model) -> str:
    img = image.load_img(filename, target_size=(300, 300))
    img = image.img_to_array(img, dtype=np.uint8)
    img = np.array(img) / 255.0
    p = model.predict(img[np.newaxis, ...])

    return DRUG_CATEGORY[np.argmax(p[0], axis=-1)]


def get_nutrition_info(nutrient: str, amount: str, food: str) -> str:
    # Replace 'YOUR_APP_ID' and 'YOUR_APP_KEY' with your actual Edamam API credentials
    app_id = '7eaaf89e'
    app_key = 'b973aef0946a4fca327396113d24ae1e'
    base_url = 'https://api.edamam.com/api/nutrition-data'

    params = {
        'app_id': app_id,
        'app_key': app_key,
        'ingr': f'{amount} {food}',
    }

    response = requests.get(base_url, params=params)
    data = json.loads(response.text)

    if data['calories'] == 0 and data['totalWeight'] == 0:
        return "Invalid input. I couldn't find any information on that food."

    # Convert nutrient to lowercase for easier comparison
    nutrient = nutrient.lower()

    nutrients_mapping = {
        'calories': 'calories',
        'protein': 'PROCNT',
        'fat': 'FAT',
        'carbs': 'CHOCDF'
    }

    if nutrient not in nutrients_mapping:
        return f"Invalid nutrient input. Please use one of the following: {', '.join(nutrients_mapping.keys())}"

    nutrient_code = nutrients_mapping[nutrient]
    nutrient_value = data['totalNutrients'][nutrient_code]['quantity']
    nutrient_unit = data['totalNutrients'][nutrient_code]['unit']

    return f"There are approximately {nutrient_value:.2f}{nutrient_unit} {nutrient} in {amount} of {food}."


#######################################################
#  Init vars
#######################################################
time.clock = time.time
read_expr = Expression.fromstring
root = tk.Tk()
root.withdraw()
model_food = load_model("model_food.h5")
model_drugs = load_model("model_drugs.h5")
food_model = YOLOv5("yolov5l.pt", device="cpu")
restaurant_data = read_csv_file("restaurants.csv")

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
print(
    "I'm here to assist you with any questions or concerns you have regarding health, wellness, and nutrition. Go ahead and ask me anything!")
print("If you would like me to speak out my replies, please enter SPEAK to start and STOP to stop")
print("If you would like to ask a question using voice, please enter VOICE.")

use_voice = False
r = sr.Recognizer()

#######################################################
# Main loop
#######################################################
engine = pyttsx3.init()
# voice properties
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # select a male voice
engine.setProperty('rate', 160)  # adjust speaking rate
engine.setProperty('volume', 0.9)  # adjust volume
engine.setProperty('pitch', 1.1)  # adjust pitch

speakout = False
while True:
    # get user input
    try:
        if not use_voice:
            user_input = input("> ")
        else:
            inputv = listen()
            if inputv is not None:
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
            reply = 'OK, I will remember that ' + object + ' is ' + subject
            print(reply)
            if speakout:
                engine.say(reply)
                engine.runAndWait()

        elif cmd == 32:  # if the input pattern is "check that * is *"
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
            print_kb()

        elif cmd == 34:
            file_path = filedialog.askopenfilename()
            pred = predict_image_food(file_path, model_food)
            print(f"This could be a {pred}")
            if speakout:
                engine.say(f"This could be a {pred}")
                engine.runAndWait()

        elif cmd == 35:
            file_path = filedialog.askopenfilename()
            pred = []
            food_names = list(FOODS_CATEGORY.values())
            pred = detect_foods(file_path, food_model, food_names)
            print("Detected foods:", pred)
            if len(pred) == 1:
                reply = f"This could be a {pred[0]}"
                print(reply)
                if speakout:
                    engine.say(f"This could be a {pred[0]}")
                    engine.runAndWait()
            elif len(pred) > 1:
                reply = "These could be "
                print(reply)
                if speakout:
                    engine.say(reply)
                    engine.runAndWait()
                for food in pred:
                    print(food)
                    if speakout:
                        engine.say(food)
                        engine.runAndWait()

        elif cmd == 36:
            file_path = filedialog.askopenfilename()
            pred = predict_image_drug(file_path, model_drugs)
            print(f"This could be a {pred}")
            if speakout:
                engine.say(f"This could be a {pred}")
                engine.runAndWait()

        elif cmd == 37:
            pred = []
            body_part = params[1]
            if body_part.lower() == 'abs':
                body_part = 'Abdominals'
            pred = get_exercises_for_body_part(body_part)
            if len(pred) == 0:
                reply = f"Sorry, I couldn't find any exercises for {body_part}. Please try again."
            else:
                reply = f"Here are 5 exercises for {body_part}: {', '.join(pred)}"
            print(reply)
            if speakout:
                engine.say(reply)
                engine.runAndWait()

        elif cmd == 38:
            nutrient, _, rest = params[1].partition(' in ')
            amount, _, food = rest.partition(' of ')
            print(nutrient, amount, food)
            reply = get_nutrition_info(nutrient, amount, food)
            print(reply)
            if speakout:
                engine.say(reply)
                engine.runAndWait()

        elif cmd == 39:
            restaurant_name = params[1]
            if restaurant_name in restaurant_data:
                service, food = restaurant_data[restaurant_name]
                low_quality, medium_quality, high_quality = evaluate_restaurant(service, food)
                crisp_quality = defuzzify(low_quality, medium_quality, high_quality)
                response = f"The crisp quality of {restaurant_name} is {crisp_quality:.2f}"
            else:
                response = f"Sorry, {restaurant_name} was not found in the dataset."
            print(response)
            if speakout:
                engine.say(response)
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
