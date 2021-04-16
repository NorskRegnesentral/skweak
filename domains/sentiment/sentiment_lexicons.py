from skweak.base import SpanAnnotator
import os
from spacy.tokens import Doc #type: ignore
from typing import Sequence, Tuple, Optional, Iterable
from collections import defaultdict

####################################################################
# Labelling sources based on lexicons
####################################################################

class LexiconAnnotator(SpanAnnotator):
    """Annotation based on a sentiment lexicon"""

    def __init__(self, name, lexicon_dir, margin=0):
        """Creates a new annotator based on a Spacy model. """
        super(LexiconAnnotator, self).__init__(name)

        self.margin = margin

        pos_file = None
        for file in os.listdir(lexicon_dir):
            if "positive" in file.lower() and "txt" in file:
                pos_file = os.path.join(lexicon_dir, file)
                self.pos = set([l.strip() for l in open(pos_file)])
        if pos_file is None:
            print("No positive lexicon file found in {}".format(lexicon_dir))

        neg_file = None
        for file in os.listdir(lexicon_dir):
            if "negative" in file.lower() and "txt" in file:
                neg_file = os.path.join(lexicon_dir, file)
                self.neg = set([l.strip() for l in open(neg_file)])
        if neg_file is None:
            print("No negative lexicon file found in {}".format(lexicon_dir))

    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:
        pos = 0
        neg = 0

        # Iterate through tokens and add up positive and negative tokens
        for token in doc:
            if token.text in self.pos:
                pos += 1
            if token.text in self.neg:
                neg += 1

        # check if there are more pos or neg tokens, plus a margin
        # Regarding labels: positive: 2, neutral: 1, negative: 0
        if pos > (neg + self.margin):
            label = 2
        elif neg > (pos + self.margin):
            label = 0
        else:
            label = 1
        yield 0, len(doc), label #type: ignore


class VADAnnotator(SpanAnnotator):
    """Annotation based on a sentiment lexicon"""

    def __init__(self, name, lexicon_path, margin=0.2):
        """Creates a new annotator based on a Spacy model. """
        super(VADAnnotator, self).__init__(name)

        self.margin = margin

        self.lexicon = defaultdict(lambda: 0.5)
        for i, line in enumerate(open(lexicon_path)):
            if i > 0: # skip the header
                en_term, no_term, v, a, d = line.strip().split("\t")
                self.lexicon[no_term] = float(v)

    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:
        scores = [0.5]

        # Iterate through tokens and add up positive and negative tokens
        for token in doc:
            scores.append(self.lexicon[token.text])

        mean_score = sum(scores) / len(scores)
        # check if there are more pos or neg tokens, plus a margin
        # Regarding labels: positive: 2, neutral: 1, negative: 0
        if mean_score > (0.5 + self.margin):
            label = 2
        elif mean_score < (0.5 + self.margin):
            label = 0
        else:
            label = 1
        yield 0, len(doc), label #type: ignore



class SocalAnnotator(SpanAnnotator):
    """Annotation based on a sentiment lexicon"""

    def __init__(self, name, lexicon_path, margin=0):
        """Creates a new annotator based on a Spacy model. """
        super(SocalAnnotator, self).__init__(name)

        self.margin = margin

        self.lexicon = defaultdict(lambda: 0)
        for i, line in enumerate(open(lexicon_path)):
            if i > 0: # skip the header
                try:
                    no_term, score = line.strip().split("\t")
                    self.lexicon[no_term] = float(score) #type: ignore
                except ValueError:
                    print(str(i) + ": " + line)

    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:
        scores = [0]

        # Iterate through tokens and add up positive and negative tokens
        for token in doc:
            scores.append(self.lexicon[token.text])

        mean_score = sum(scores) / len(scores)
        # check if there are more pos or neg tokens, plus a margin
        # Regarding labels: positive: 2, neutral: 1, negative: 0
        if mean_score > (0 + self.margin):
            label = 2
        elif mean_score < (0 + self.margin):
            label = 0
        else:
            label = 1
        yield 0, len(doc), label #type: ignore


class NRC_SentAnnotator(SpanAnnotator):
    """Annotation based on a sentiment lexicon"""

    def __init__(self, name, lexicon_path, margin=0):
        """Creates a new annotator based on a Spacy model. """
        super(NRC_SentAnnotator, self).__init__(name)

        self.margin = margin
        self.pos = set()
        self.neg = set()

        for i, line in enumerate(open(lexicon_path)):
            try:
                no_term, sent, score = line.strip().split("\t")
                if int(score) == 1:
                    if sent == "positive":
                        self.pos.add(no_term)
                    if sent == "negative":
                        self.neg.add(no_term)
            except:
                pass

    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:
        pos = 0
        neg = 0

        # Iterate through tokens and add up positive and negative tokens
        for token in doc:
            if token.text in self.pos:
                pos += 1
            if token.text in self.neg:
                neg += 1

        # check if there are more pos or neg tokens, plus a margin
        # Regarding labels: positive: 2, neutral: 1, negative: 0
        if pos > (neg + self.margin):
            label = 2
        elif neg > (pos + self.margin):
            label = 0
        else:
            label = 1
        yield 0, len(doc), label #type: ignore


class BUTAnnotator(SpanAnnotator):
    """Annotation based on the heuristic"""

    def __init__(self, name, lexicon_dir, margin=0):
        """Creates a new annotator based on a Spacy model. """
        super(BUTAnnotator, self).__init__(name)

        self.margin = margin

        pos_file = None
        for file in os.listdir(lexicon_dir):
            if "positive" in file.lower() and "txt" in file:
                pos_file = os.path.join(lexicon_dir, file)
                self.pos = set([l.strip() for l in open(pos_file)])
        if pos_file is None:
            print("No positive lexicon file found in {}".format(lexicon_dir))

        neg_file = None
        for file in os.listdir(lexicon_dir):
            if "negative" in file.lower() and "txt" in file:
                neg_file = os.path.join(lexicon_dir, file)
                self.neg = set([l.strip() for l in open(neg_file)])
        if neg_file is None:
            print("No negative lexicon file found in {}".format(lexicon_dir))

    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:
        pos = 0
        neg = 0

        # Iterate through tokens and add up positive and negative tokens
        tokens = [t.text for t in doc]
        if "men" in tokens:
            idx = tokens.index("men") + 1
            for token in tokens[idx:]:
                if token in self.pos:
                    pos += 1
                if token in self.neg:
                    neg += 1

        # check if there are more pos or neg tokens, plus a margin
        # Regarding labels: positive: 2, neutral: 1, negative: 0
        if pos > (neg + self.margin):
            label = 2
        elif neg > (pos + self.margin):
            label = 0
        else:
            label = 1
        yield 0, len(doc), label #type: ignore
