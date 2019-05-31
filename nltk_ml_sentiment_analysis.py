import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import random

print("Machine-Learning----------------------------------------------------------")
# Step 1 – Training data
train = [("Great place to be when you are in Granada.", "pos"),
  ("The place was being renovated when I visited so the seating was limited.", "neg"),
  ("Loved the ambience, loved the food", "pos"),
  ("The food is delicious but not over the top.", "neg"),
  ("Service - Little slow, probably because too many people.", "neg"),
  ("The place is not easy to locate", "neg"),
  ("Mushroom fried rice was spicy", "pos"),
  (":) and :D", "pos"),
  (":*", "pos"),
  ("AWESOME place...delicious food!!!", "pos")
  ]
  
# Step 2
dictionary = set(word.lower() for passage in train for word in word_tokenize(passage[0]))
print(dictionary)

# Step 3
t = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in train]
 
# Step 4 – the classifier is trained with sample data
print("Naive-Bayes----------------------------------------------------------")
classifier = nltk.NaiveBayesClassifier.train(t)
print (classifier.show_most_informative_features())
  
test_data_en = "Paella was hot and spicy"
print(test_data_en)
test_data_features_en = {word.lower(): (word in word_tokenize(test_data_en.lower())) for word in dictionary}
print (classifier.classify(test_data_features_en))

test_data_es = "Nice and Beautiful!"
print(test_data_es)
test_data_features_es = {word.lower(): (word in word_tokenize(test_data_es.lower())) for word in dictionary}
print (classifier.classify(test_data_features_es))



