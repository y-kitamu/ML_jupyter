from sklearn.feature_extraction.text import HashingVectorizer
import re
from pathlib import Path
import pickle

cur_path = Path(__file__).parent
stop = pickle.load(
    (cur_path / Path('pkl_objects') / Path('stopwords.pkl')).open('rb'))

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    test = re.sub('[\W]+', ' ', text.lower()) \
        + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)
