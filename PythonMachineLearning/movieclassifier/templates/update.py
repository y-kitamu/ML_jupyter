import pickle
import sqlite3
import numpy as np
from pathlib import Path

from vectorizer import vect

def update_model(db_path, model, batch_size=10000):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT * from review_db')

    results = c.fetchmany(batch_size)
    while results:
        data = np.array(results)
        X = data[:, 0]
        y = data[:, 1].astype(int)
        classes = np.array([0, 1])
        X = vect.transform(X)
        model.partial_fit(X_train, y, classes=classes)
        results = c.fetchmany(batch_size)

    conn.cose()
    return model

cur_dir = Path(__file__).parent
clf = pickle.load((cur_dir / Path('pkl_objects', 'classifier.pkl')).open('rb'))

db = cur_dir / Path('reviews.sqlite')

clf = update_model(db_path=str(db), model=clf)

pickle.dump(clf, (cur_dir / Path('pkl_objects', 'classifier.pkl')).open('wb'),
            protocol=4)
