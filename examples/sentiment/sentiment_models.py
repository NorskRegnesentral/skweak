from skweak.base import SpanAnnotator
import os
from spacy.tokens import Doc # type: ignore
from typing import Sequence, Tuple, Optional, Iterable
from collections import defaultdict

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from transformers import pipeline, BertForSequenceClassification, BertTokenizer

import tarfile
import pickle
import os


class MBertAnnotator(SpanAnnotator):
    """Annotation based on multi-lingual BERT trained on Stanford Sentiment Treebank"""
    def __init__(self, name):
        super(MBertAnnotator, self).__init__(name)
        self.classifier = BertForSequenceClassification.from_pretrained("../data/sentiment/models/sst", num_labels=3)
        self.classifier.eval() # type: ignore
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
        print("Loaded mBERT from {}".format("../data/sentiment/models/sst"))

    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:

        text = [" ".join([t.text for t in doc])]
        encoding = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        output = self.classifier(**encoding)
        # classifier outputs a dict, eg {'label': '5 stars', 'score': 0.99}
        # so we need to get the label and transform it to an int
        _, p = output.logits.max(1)
        label = int(p[0])
        yield 0, len(doc), label # type: ignore


class MultilingualAnnotator(SpanAnnotator):
    """Annotation based on multi-lingual BERT trained on review data in 6 languages"""

    def __init__(self, name):
        """Creates a new annotator based on a Spacy model. """
        super(MultilingualAnnotator, self).__init__(name)

        self.classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
        print("Loaded nlptown/bert-base-multilingual-uncased-sentiment")

    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:

        text = [" ".join([t.text for t in doc])]
        labels = self.classifier(text)[0]
        # classifier outputs a dict, eg {'label': '5 stars', 'score': 0.99}
        # so we need to get the label and transform it to an int
        pred = int(labels["label"][0])

        # check if there are more pos or neg tokens, plus a margin
        # Regarding labels: positive: 2, neutral: 1, negative: 0
        if pred > 3:
            label = 2
        elif pred < 3:
            label = 0
        else:
            label = 1
        yield 0, len(doc), label # type: ignore


class DocBOWAnnotator(SpanAnnotator):
    """Annotation based on a TF-IDF Bag-of-words document-level classifier"""

    def __init__(self, name, model_path, doclevel_data=None):
        """Creates a new annotator based on a Spacy model. """
        super(DocBOWAnnotator, self).__init__(name)

        self.model_path = model_path
        self.doclevel_data = doclevel_data

        if self.doclevel_data is not None:
            print("Fitting model on {}".format(self.doclevel_data))
            self.fit(doclevel_data)
            print("Saving vectorizer and model to {}".format(model_path))
            self.save_model(self.model_path)
        else:
            try:
                self.load_model(self.model_path)
                print("Loaded model from {}".format(self.model_path))
            except FileNotFoundError:
                print("Trained model not found. Train a model first by providing the doclevel_data when instantiating the annotator.")

    def save_model(self, model_path):
        os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, "vectorizer.pkl"), "wb") as o:
            pickle.dump(self.vectorizer, o)

        with open(os.path.join(model_path, "bow_model.pkl"), "wb") as o:
            pickle.dump(self.model, o)

    def load_model(self, model_path):
        with open(os.path.join(model_path, "vectorizer.pkl"), "rb") as o:
            self.vectorizer = pickle.load(o)
        with open(os.path.join(model_path, "bow_model.pkl"), "rb") as o:
            self.model = pickle.load(o)

    def open_norec_doc(self, file_path, split="train"):
        tar = tarfile.open(file_path, "r:gz")

        train_names = [tarinfo for tarinfo in tar.getmembers() if split in tarinfo.name and ".conllu" in tarinfo.name]

        docs, ratings = [], []

        for fname in train_names:
            content = tar.extractfile(fname)
            language = content.readline().decode("utf8").rstrip("\n")[-2:]
            rating = content.readline().decode("utf8").rstrip("\n")[-1]
            doc_id = content.readline().decode("utf8").rstrip("\n").split()[-1]

            words = []
            for line in content:
                line = line.decode("utf8")
                if line[0] == '#':
                    continue
                if not line.rstrip("\n"):
                    continue
                else:
                    words.append(line.split("\t")[1])

            docs.append(" ".join(words))
            ratings.append(int(rating))
        return docs, ratings

    def fit(self, file_path):
        train_docs, train_ratings = self.open_norec_doc(file_path, split="train")
        test_docs, test_ratings = self.open_norec_doc(file_path, split="test")

        self.vectorizer = TfidfVectorizer()
        trainX = self.vectorizer.fit_transform(train_docs)
        self.model = LinearSVC()
        self.model.fit(trainX, train_ratings)

        testX = self.vectorizer.transform(test_docs)

        pred = self.model.predict(testX)
        print("Doc-level F1: {0:.3f}".format(f1_score(test_ratings, pred, average="macro")))


    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:

        text = [" ".join([t.text for t in doc])]
        X = self.vectorizer.transform(text)
        pred = self.model.predict(X)[0]

        # check if there are more pos or neg tokens, plus a margin
        # Regarding labels: positive: 2, neutral: 1, negative: 0
        if pred > 4:
            label = 2
        elif pred < 3:
            label = 0
        else:
            label = 1
        yield 0, len(doc), label

