
from abc import abstractmethod
import itertools
from typing import Sequence, Tuple, Optional, Iterable
from . import utils
from spacy.tokens import Doc, Span  # type: ignore


############################################
# Abstract class for all annotators
############################################

class AbstractAnnotator:
    """Base class for all annotation or aggregation sources 
    employed in skweak"""

    def __init__(self, name: str):
        """Initialises the annotator with a name"""
        self.name = name

    @abstractmethod
    def __call__(self, doc: Doc) -> Doc:
        """Annotates a single Spacy Doc object"""

        raise NotImplementedError()

    def pipe(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        """Annotates a stream of Spacy Doc objects"""

        # This is the default implementation, which should be replaced if
        # we have better ways of annotating large numbers of documents
        for doc in docs:
            yield self(doc)

    def annotate_docbin(self, docbin_input_path: str,
                        docbin_output_path: Optional[str] = None,
                        spacy_model_name: str = "en_core_web_md",
                        cutoff: Optional[int] = None, nb_to_skip: int = 0):
        """Runs the annotator on the documents of a DocBin file, and write the output
        to docbin_output_path (or to the same file if it is set to None). The spacy 
        model name must be the same as the one used to create the DocBin file in the 
        first place. 

        If cutoff is set, the annotation stops after the given number of documents. If
        nb_to_skip is set, the method skips a number of documents at the start.
        """

        docs = utils.docbin_reader(docbin_input_path, spacy_model_name,
                                   cutoff=cutoff, nb_to_skip=nb_to_skip)
        new_docs = []
        for doc in self.pipe(docs):
            new_docs.append(doc)
            if len(new_docs) % 1 == 0:
                print("Number of processed documents:", len(new_docs))

        docbin_output_path = docbin_output_path or docbin_input_path
        utils.docbin_writer(new_docs, docbin_output_path)


####################################################################
# Type of annotators
####################################################################

class SpanAnnotator(AbstractAnnotator):
    """Generic class for the annotation of token spans"""

    def __init__(self, name: str):
        """Initialises the annotator with a source name"""

        super(SpanAnnotator, self).__init__(name)

        # Set of other labelling sources that have priority
        self.incompatible_sources = []

    # type:ignore
    def add_incompatible_sources(self, other_sources: Sequence[str]):
        """Specifies a list of sources that are not compatible with the current 
        source and should take precedence over it in case of overlap"""

        self.incompatible_sources.extend(other_sources)

    def __call__(self, doc: Doc) -> Doc:

        # We start by clearing all existing annotations
        doc.spans[self.name] = []

        # And we look at all suggested spans
        for start, end, label in self.find_spans(doc):

            # We only add the span if it is compatible with other sources
            if self._is_allowed_span(doc, start, end):
                span = Span(doc, start, end, label)
                doc.spans[self.name].append(span)

        return doc

    @abstractmethod
    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:
        """Generates (start, end, label) triplets corresponding to token-level
        spans associated with a given label. """

        raise NotImplementedError("Must implement find_spans method")

    def _is_allowed_span(self, doc, start, end):
        """Checks whether the span is allowed (given incompatibilities with other sources)"""

        for other_source in self.incompatible_sources:

            intervals = sorted((span.start, span.end) for span in
                               doc.spans.get(other_source, []))

            # Performs a binary search to efficiently detect overlapping spans
            start_search, end_search = utils._binary_search(
                start, end, intervals)
            for interval_start, interval_end in intervals[start_search:end_search]:
                if start < interval_end and end > interval_start:
                    return False
        return True


####################################################################
# Combination of annotators
####################################################################


class CombinedAnnotator(AbstractAnnotator):
    """Annotator of entities in documents, combining several sub-annotators  """

    def __init__(self):
        super(CombinedAnnotator, self).__init__("")
        self.annotators = []

    def __call__(self, doc: Doc) -> Doc:
        """Annotates a single  document with the sub-annotators
        NB: avoid using this method for large collections of documents (as it is quite 
        inefficient), and prefer the method pipe that runs on batches of documents.
        """

        for annotator in self.annotators:
            doc = annotator(doc)
        return doc

    def pipe(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        """Annotates the stream of documents using the sub-annotators."""

        # We duplicate the streams of documents
        streams = itertools.tee(docs, len(self.annotators)+1)

        # We create one pipe per annotator
        pipes = [annotator.pipe(stream) for annotator, stream in
                 zip(self.annotators, streams[1:])]

        for doc in streams[0]:
            for pipe in pipes:
                try:
                    next(pipe)
                except BaseException as e:
                    print("ignoring document:", doc)
                    raise e

            yield doc

    def add_annotator(self, annotator: AbstractAnnotator):
        """Adds an annotator to the list"""

        self.annotators.append(annotator)
        return self

    def add_annotators(self, *annotators: AbstractAnnotator):
        """Adds several annotators to the list"""

        for annotator in annotators:
            self.add_annotator(annotator)
        return self

    def get_annotator(self, annotator_name: str):
        """Returns the annotator identified by its name (and throws an
        exception if no annotator can be found)"""

        for annotator in self.annotators:
            if annotator.name == annotator_name:
                return annotator

        raise RuntimeError("Could not find annotator %s" % annotator_name)
