import pandas as pd
from tqdm import tqdm
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import matplotlib.pyplot as plt
import nltk
import multiprocessing
tqdm.pandas(desc="progress-bar")
cores = multiprocessing.cpu_count()
plt.style.use('ggplot')

if __name__=='__main__':
    # Read in the data, dropping nulls and columns that won't help predictions
    df = pd.read_csv('../data/train-balanced-sarcasm.csv')
    df.drop(['author','ups','downs','date'], axis=1, inplace=True)
    df.dropna(inplace=True)

    # Re-index our dataframe after dropping nulls
    df.shape
    df.index = range(1010773)

    # Train-Test split our data in a 70-30 split, then tokenize the comment and parents comment columns
    train, test = train_test_split(df, test_size=0.3, random_state=42)

    def tokenize_text(text):
        tokens = []
        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                if len(word) < 2:
                    continue
                tokens.append(word.lower())
        return tokens

    train_tagged = train.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['comment']), tags=[r.comment]), axis=1)
    test_tagged = test.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['comment']), tags=[r.comment]), axis=1)
    train_tagged = train.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['parent_comment']), tags=[r.parent_comment]), axis=1)
    test_tagged = test.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['parent_comment']), tags=[r.parent_comment]), axis=1)

    # Build our vocab
    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
    model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

    # Train our doc2vec model in gensim with 30 epochs
    for epoch in range(10):
        model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    def vec_for_learning(model, tagged_docs):
        sents = tagged_docs.values
        targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
        return targets, regressors

    # Attempt a logistic regression on our data
    y_train, X_train = vec_for_learning(model_dbow, train_tagged)
    y_test, X_test = vec_for_learning(model_dbow, test_tagged)

    logreg = LogisticRegression(n_jobs=1, C=1e5)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    from sklearn.metrics import accuracy_score, f1_score

    print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
    print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
