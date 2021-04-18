

from skweak.base import SpanAnnotator
import spacy
import itertools
import json
from spacy.tokens import Doc, Span  # type: ignore
from typing import Tuple, Iterable, List

####################################################################
# Labelling source based on neural models
####################################################################


class ModelAnnotator(SpanAnnotator):
    """Annotation based on a spacy NER model"""

    def __init__(self, name:str, model_path:str, 
                 disabled:List[str]=["parser", "tagger", "lemmatizer", "attribute_ruler"]):
        """Creates a new annotator based on a Spacy model. """
        
        super(ModelAnnotator, self).__init__(name)
        self.model = spacy.load(model_path, disable=disabled)


    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:
        """Annotates one single document using the Spacy NER model"""

        # Create a new document (to avoid conflicting annotations)
        doc2 = self.create_new_doc(doc)
        # And run the model
        for _, proc in self.model.pipeline:
            doc2 = proc(doc2)
        # Add the annotation
        for ent in doc2.ents:
            yield ent.start, ent.end, ent.label_

    def pipe(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        """Annotates the stream of documents based on the Spacy model"""

        stream1, stream2 = itertools.tee(docs, 2)

        # Remove existing entities from the document
        stream2 = (self.create_new_doc(d) for d in stream2)
        
        # And run the model
        for _, proc in self.model.pipeline:
            stream2 = proc.pipe(stream2)
        
        for doc, doc_copy in zip(stream1, stream2):

            doc.spans[self.name] = []

            # Add the annotation
            for ent in doc_copy.ents:
                doc.spans[self.name].append(Span(doc, ent.start, ent.end, ent.label_))

            yield doc

    def create_new_doc(self, doc: Doc) -> Doc:
        """Create a new, empty Doc (but with the same tokenisation as before)"""

        return spacy.tokens.Doc(self.model.vocab, [tok.text for tok in doc], #type: ignore
                               [tok.whitespace_ for tok in doc])


class TruecaseAnnotator(ModelAnnotator):
    """Spacy model annotator that preprocess all texts to convert them to a 
    "truecased" representation (see below)"""

    def __init__(self, name:str, model_path:str, form_frequencies:str,
                 disabled:List[str]=["parser", "tagger", "lemmatizer", "attribute_ruler"]):
        """Creates a new annotator based on a Spacy model, and a dictionary containing
        the most common case forms for a given word (to be able to truecase the document)."""
        
        super(TruecaseAnnotator, self).__init__(name, model_path, disabled)
        with open(form_frequencies) as fd:
            self.form_frequencies = json.load(fd)

    def create_new_doc(self, doc: Doc, min_prob: float = 0.25) -> Doc:
        """Performs truecasing of the tokens in the spacy document. Based on relative 
        frequencies of word forms, tokens that 
        (1) are made of letters, with a first letter in uppercase
        (2) and are not sentence start
        (3) and have a relative frequency below min_prob
        ... will be replaced by its most likely case (such as lowercase). """

  #      print("running on", doc[:10])

        if not self.form_frequencies:
            raise RuntimeError(
                "Cannot truecase without a dictionary of form frequencies")

        tokens = []
        spaces = []
        doctext = doc.text
        for tok in doc:
            toktext = tok.text

            # We only change casing for words in Title or UPPER
            if tok.is_alpha and toktext[0].isupper():
                cond1 = tok.is_upper and len(toktext) > 2  # word in uppercase
                cond2 = toktext[0].isupper(
                ) and not tok.is_sent_start  # titled word
                if cond1 or cond2:
                    token_lc = toktext.lower()
                    if token_lc in self.form_frequencies:
                        frequencies = self.form_frequencies[token_lc]
                        if frequencies.get(toktext, 0) < min_prob:
                            alternative = sorted(
                                frequencies.keys(), key=lambda x: frequencies[x])[-1]

                            # We do not change from Title to to UPPER
                            if not tok.is_title or not alternative.isupper():
                                toktext = alternative

            tokens.append(toktext)

            # Spacy needs to know whether the token is followed by a space
            if tok.i < len(doc)-1:
                spaces.append(doctext[tok.idx+len(tok)].isspace())
            else:
                spaces.append(False)

        # Creates a new document with the tokenised words and space information
        doc2 = Doc(self.model.vocab, words=tokens, spaces=spaces) #type: ignore
 #       print("finished with doc", doc2[:10])
        return doc2
