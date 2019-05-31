import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

print("Naive-Bayes-pre-labeled----------------------------------------------------------")

# step 1
def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})

print(format_sentence("The data science is very sexy"))

# step 2
pos = []
with open("../pos_tweets.txt") as f:
    for i in f: 
        pos.append([format_sentence(i), 'pos'])

neg = []
with open("../neg_tweets.txt") as f:
    for i in f: 
        neg.append([format_sentence(i), 'neg'])

# Step 3
# next, split labeled data into the training and test data
training = pos[:int((.8)*len(pos))] + neg[:int((.8)*len(neg))]
test = pos[int((.8)*len(pos)):] + neg[int((.8)*len(neg)):]

#Step 4
classifier = NaiveBayesClassifier.train(training)
classifier.show_most_informative_features()

print("---------------------------------------------------------------")
example1 = "Opinions are awesome!"
print(example1)
print(classifier.classify(format_sentence(example1)))

example2 = "I don't like maths :("
print(example2)
print(classifier.classify(format_sentence(example2)))

print("---------------------------------------------------------------")

print(accuracy(classifier, test))

