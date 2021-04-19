import sys
import spacy
from spacy.tokens import DocBin
import pandas as pd

from .norec_sentiment import FullSentimentAnnotator
from skweak import utils
from sklearn.metrics import f1_score
from .sentiment_models import MBertAnnotator

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

import skweak


##################################################################
# Preprocessing
##################################################################

nlp = spacy.load("nb_core_news_md")

train_doc_bin = DocBin(store_user_data=True)
dev_doc_bin = DocBin(store_user_data=True)
test_doc_bin = DocBin(store_user_data=True)

train = pd.read_csv("./data/sentiment/norec_sentence/train.txt", delimiter="\t", header=None) #type: ignore
dev = pd.read_csv("./data/sentiment/norec_sentence/dev.txt", delimiter="\t", header=None) #type: ignore
test = pd.read_csv("./data/sentiment/norec_sentence/test.txt", delimiter="\t", header=None) #type: ignore

for sid, (label, sent) in train.iterrows():
    doc = nlp(sent)
    doc.user_data["gold"] = label
    train_doc_bin.add(doc)
train_doc_bin.to_disk("./data/sentiment/norec_sentence/train.docbin")

for sid, (label, sent) in dev.iterrows():
    doc = nlp(sent)
    doc.user_data["gold"] = label
    dev_doc_bin.add(doc)
dev_doc_bin.to_disk("./data/sentiment/norec_sentence/dev.docbin")

for sid, (label, sent) in test.iterrows():
    doc = nlp(sent)
    doc.user_data["gold"] = label
    test_doc_bin.add(doc)
test_doc_bin.to_disk("./data/sentiment/norec_sentence/test.docbin")


##################################################################
# Weak supervision
##################################################################

ann = FullSentimentAnnotator()
ann.add_all()

ann.annotate_docbin("./data/sentiment/norec_sentence/train.docbin", "./data/sentiment/norec_sentence/train_pred.docbin")

ann.annotate_docbin("./data/sentiment/norec_sentence/dev.docbin", "./data/sentiment/norec_sentence/dev_pred.docbin")

ann.annotate_docbin("./data/sentiment/norec_sentence/test_pred.docbin", "./data/sentiment/norec_sentence/test_pred.docbin")

unified_model = skweak.aggregation.HMM("hmm", [0, 1, 2], sequence_labelling=False) #type: ignore
unified_model.fit("./data/sentiment/norec_sentence/train_pred.docbin")
unified_model.annotate_docbin("./data/sentiment/norec_sentence/train_pred.docbin", "./data/sentiment/norec_sentence/train_pred.docbin")

#unified_model = skweak.aggregation.HMM("hmm", [0, 1, 2], sequence_labelling=False)
#unified_model.fit("./data/sentiment/norec_sentence/dev_pred.docbin")
unified_model.annotate_docbin("./data/sentiment/norec_sentence/dev_pred.docbin", "./data/sentiment/norec_sentence/dev_pred.docbin")

#unified_model = skweak.aggregation.HMM("hmm", [0, 1, 2], sequence_labelling=False)
#unified_model.fit("./data/sentiment/norec_sentence/test_pred.docbin")
unified_model.annotate_docbin("./data/sentiment/norec_sentence/test_pred.docbin", "./data/sentiment/norec_sentence/test_pred.docbin")

mv = skweak.aggregation.MajorityVoter("mv", [0, 1, 2], sequence_labelling=False) #type: ignore
mv.annotate_docbin("./data/sentiment/norec_sentence/test_pred.docbin", "./data/sentiment/norec_sentence/test_pred.docbin")

pred_docs = list(utils.docbin_reader("./data/sentiment/norec_sentence/test_pred.docbin"))


##################################################################
# Evaluation of upper bound
##################################################################


train_docs = list(utils.docbin_reader("./data/sentiment/norec_sentence/train.docbin"))

pred_docs = list(utils.docbin_reader("./data/sentiment/norec_sentence/test_pred.docbin"))

vectorizer = TfidfVectorizer(ngram_range=(1, 3))
model = LinearSVC()

train = [" ".join([t.text for t in doc]) for doc in train_docs]
trainX = vectorizer.fit_transform(train)
train_y = [doc.user_data["gold"] for doc in train_docs]
model.fit(trainX, train_y)

test = [" ".join([t.text for t in doc]) for doc in pred_docs]
testX = vectorizer.transform(test)
pred = model.predict(testX)

gold = [d.user_data["gold"] for d in pred_docs]

f1 = f1_score(gold, pred, average="macro")
print("Upper Bound F1: {0:.3f}".format(f1))

##################################################################
# Evaluation of majority baseline
##################################################################

maj_class = [1] * len(gold)
maj_f1 = f1_score(gold, maj_class, average="macro")
print("Majority class: {0:.3f}".format(maj_f1))

print("-" * 25)

##################################################################
# Evaluation of labelling functions
##################################################################


for lexicon in pred_docs[0].user_data["spans"].keys():
    pred = []
    for d in pred_docs:
        for span in d.spans[lexicon]:
            pred.append(span.label_)

    lex_f1 = f1_score(gold, pred, average="macro")
    print("{0}:\t{1:.3f}".format(lexicon, lex_f1))

##################################################################
# Evaluation of aggregating functions
##################################################################



for aggregator in ["mv", "hmm"]:
    pred = []
    for d in pred_docs:
        for span in d.spans[aggregator]:
            pred.append(span.label_)
    hmm_f1 = f1_score(gold, pred, average="macro")
    print("{0}:\t{1:.3f}".format(aggregator, hmm_f1))

