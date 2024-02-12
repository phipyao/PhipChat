import discord
import os
from discord.ext import commands
from dotenv import load_dotenv 
load_dotenv()
import random 
import json 
import pickle 
import numpy as np 
import nltk 
from keras.models import load_model 
from nltk.stem import WordNetLemmatizer 

intents = discord.Intents().all()
client = commands.Bot(command_prefix="!", intents=intents)

lemmatizer = WordNetLemmatizer() 

# load files
intents = json.loads(open("intense.json").read()) 
words = pickle.load(open('words.pkl', 'rb')) 
classes = pickle.load(open('classes.pkl', 'rb')) 
model = load_model('chatbotmodel.h5') 

def clean_up_sentences(sentence): 
	sentence_words = nltk.word_tokenize(sentence) 
	sentence_words = [lemmatizer.lemmatize(word) 
					for word in sentence_words] 
	return sentence_words 

def bagw(sentence): 	
	# separate out words from the input sentence 
	sentence_words = clean_up_sentences(sentence) 
	bag = [0]*len(words) 
	for w in sentence_words: 
		for i, word in enumerate(words): 
			# check whether the word is present in the input as well 
			if word == w: 
				bag[i] = 1
	return np.array(bag) 

def predict_class(sentence): 
	bow = bagw(sentence) 
	res = model.predict(np.array([bow]))[0] 
	ERROR_THRESHOLD = 0.25
	results = [[i, r] for i, r in enumerate(res) 
			if r > ERROR_THRESHOLD] 
	results.sort(key=lambda x: x[1], reverse=True) 
	return_list = [] 
	for r in results: 
		return_list.append({'intent': classes[r[0]], 
							'probability': str(r[1])}) 
		return return_list 

def get_response(intents_list, intents_json): 
	tag = intents_list[0]['intent'] 
	list_of_intents = intents_json['intents'] 
	result = "" 
	for i in list_of_intents: 
		if i['tag'] == tag: 
			# prints a random response
			result = random.choice(i['responses']) 
			break
	return result 

def chat(user_input):
    ints = predict_class(user_input) 
    res = get_response(ints, intents) 
    return res

@client.event
async def on_ready():
    print("phipbot is up!")

@client.event
async def on_message(message):
    if message.author == client.user:
        return    
    answer = chat(message.content)
    await message.channel.send(answer)

@client.command()
@commands.is_owner()
async def shutdown(context):
    exit()

TOKEN = os.getenv('DISCORD_TOKEN')
client.run(TOKEN)