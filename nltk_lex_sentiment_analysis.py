import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import json

print("Lexicons-With-VADER----------------------------------------------------------")  
restaurant_revision = ["Great place to be when you are in Granada.",
"The place was being renovated when I visited so the seating was limited.",
"Loved the ambience, loved the food",
"The food is delicious but not over the top.",
"Service - Little slow, probably because too many people.",
"The place is not easy to locate",
"Mushroom fried rice was tasty",
":) and :D",
":*",
"El lugar es bonito, pero como no sé hablar en inglés...nadie me entiendo :(",
"Vaya si que me entienden!!! El lugar es bonito y se come delicioso...",
"Fue un falso positivo, ya sé: just translate it"]
  
sid = SentimentIntensityAnalyzer()
for sentence in restaurant_revision:
     print(sentence)
     ss = sid.polarity_scores(sentence)
     for k in ss:
         print('{0}: {1}, '.format(k, ss[k]), end='')
     print()

print("No-English----------------------------------------------------------")
languages = ["English", "French", "German", "Spanish", "Italian", "Russian", "Japanese", "Arabic", "Chinese"]
language_codes = ["en", "fr", "de", "es", "it", "ru", "ja", "ar", "zh"]
nonEnglish_sentences = ["I'm surprised to see just how amazingly helpful VADER is!",
                                "Je suis surpris de voir juste comment incroyablement utile VADER est!",
                                "Ich bin überrascht zu sehen, nur wie erstaunlich nützlich VADER!",
                                "Me sorprende ver sólo cómo increíblemente útil VADER!",
                                "Sono sorpreso di vedere solo come incredibilmente utile VADER è!",
                                "Я удивлен увидеть, как раз как удивительно полезно ВЕЙДЕРА!",
                                "私はちょうどどのように驚くほど役に立つベイダーを見て驚いています!",
                                "أنا مندهش لرؤية فقط كيف مثير للدهشة فيدر فائدة!",
                                "惊讶地看到有用维德是的只是如何令人惊讶了 ！"
                                ]
for sentence in nonEnglish_sentences:
     to_lang = "en"
     from_lang = language_codes[nonEnglish_sentences.index(sentence)]
     if (from_lang == "en") or (from_lang == "en-US"):
           translation = sentence
           translator_name = "No translation needed"
     else:  # http://mymemory.translated.net/doc/usagelimits.php
            # MY MEMORY NET   http://mymemory.translated.net
           api_url = "http://mymemory.translated.net/api/get?q={}&langpair={}|{}".format(sentence, from_lang,
                                                                                              to_lang)
           hdrs = {
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                    'Accept-Encoding': 'none',
                    'Accept-Language': 'en-US,en;q=0.8',
                    'Connection': 'keep-alive'}
           response = requests.get(api_url, headers=hdrs)
           response_json = json.loads(response.text)
           translation = response_json["responseData"]["translatedText"]
           translator_name = "MemoryNet Translation Service"
     vs = sid.polarity_scores(translation)
     print("- {: <8}: {: <69} ({})".format(languages[nonEnglish_sentences.index(sentence)], sentence, translator_name))
     for k in vs:
         print('{0}: {1}, '.format(k, vs[k]), end='')
     print()
     
print()

print("MachineLearning----------------------------------------------------------")

