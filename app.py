import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import pickle

nlp = spacy.load("en_core_web_sm")

sp = spacy.load('en_core_web_sm')
all_stopwords = sp.Defaults.stop_words
all_stopwords.add('&')
all_stopwords.add(',')
all_stopwords.add('.')
all_stopwords.add('@')
all_stopwords.add('/')
all_stopwords.add(':')
all_stopwords.add('?')

data = pd.read_csv('./datasets/products_catalog.csv')
df = data.copy()
print(df.head())


def remove_stopwords_punctuation(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]
    return " ".join(tokens_without_sw)


df["ProductName"] = df["ProductName"].apply(lambda x: remove_stopwords_punctuation(x))
df["Description"] = df["Description"].apply(lambda x: remove_stopwords_punctuation(x))
df["Feature_Set"] = df["ProductBrand"] + df["ProductName"] + df["Gender"] + df["Description"] + df["PrimaryColor"]

subset = df[["ProductID", "Feature_Set"]]
subset.dropna(axis=0, inplace=True)
subset.reset_index(drop=True, inplace=True)

model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_embeddings = model.encode(subset["Feature_Set"])
print("Sentence embeddings Shape: ", sentence_embeddings.shape)

# Save model using pickle
with open('models/model_pickle.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/sentence_embeddings.pkl', 'wb') as f:
    pickle.dump(sentence_embeddings, f)

with open('models/subset.pkl', 'wb') as f:
    pickle.dump(subset, f)

with open('models/dataframe.pkl', 'wb') as f:
    pickle.dump(df, f)
