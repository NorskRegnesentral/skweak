

from typing import Callable, Collection, Dict, Iterable, Optional, Sequence, Tuple
from spacy.tokens import Span, Token, Doc  # type: ignore
from .base import SpanAnnotator

####################################################################
# Labelling sources based on heuristics / handcrafted rules
####################################################################


class FunctionAnnotator(SpanAnnotator):
    """Annotation based on a heuristic function that generates (start,end,label)
    given a spacy document"""

    def __init__(self, name: str, function: Callable[[Doc], Doc],
                 to_exclude: Sequence[str] = ()):
        """Create an annotator based on a function generating labelled spans given 
        a Spacy Doc object. Spans that overlap with existing spans from sources 
        listed in 'to_exclude' are ignored. """

        super(FunctionAnnotator, self).__init__(name)
        self.find_spans = function
        self.add_incompatible_sources(to_exclude)


class TokenConstraintAnnotator(SpanAnnotator):
    """Annotator relying on a token-level constraint. Continuous spans that 
    satisfy this constraint will be marked by the provided label."""

    def __init__(self, name: str, constraint: Callable[[Token], bool],
                 label: str, min_characters=3):
        """Given a token-level constraint, a label name, and a minimum
        number of characters, annotates with the label all (maximal) 
        contiguous spans whose tokens satisfy the constraint."""

        super(TokenConstraintAnnotator, self).__init__(name)
        self.constraint = constraint
        self.label = label
        self.min_characters = min_characters
        self.gap_tokens = {"-"}  # Hyphens should'nt stop a span

    def add_gap_tokens(self, gap_tokens: Collection[str]):
        """Adds tokens (typically function words) that are allowed in the span 
        even if they do not satisfy the constraint, provided they are surrounded
        by words that do satisfy the constraint. """

        self.gap_tokens.update(gap_tokens)

    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:
        """
        Searches for all spans whose tokens satisfy the constraint (and meet
        the minimum character length), and marks those with the provided label. 
        """

        i = 0
        while i < len(doc):
            tok = doc[i]
            # We search for the longest span that satisfy the constraint
            if self.constraint(tok):
                j = i+1
                while j < len(doc):
                    # We check the constraint
                    if self.constraint(doc[j]):
                        j += 1

                    # We also check whether the token is a gap word
                    elif (doc[j].text in self.gap_tokens and j < len(doc)-1
                          and self.constraint(doc[j+1])):
                        j += 2
                    else:
                        break

                # We check whether the span has a minimal length
                if len(doc[i:j].text) >= self.min_characters:
                    yield i, j, self.label

                i = j
            else:
                i += 1


class SpanConstraintAnnotator(SpanAnnotator):
    """Annotation by looking at text spans (from another source) 
    that satisfy a span-level constraint"""

    def __init__(self, name: str, other_name: str, constraint: Callable[[Span], bool],
                 label: Optional[str] = None):
        """Creates a new annotator that looks at the annotations from the
        other_name source, and adds them to this source if it satisfied a 
        given constraint on spans. If label is other than None, the method
        simply reuses the same label as the one specified by other_name."""

        super(SpanConstraintAnnotator, self).__init__(name)
        self.other_name = other_name
        self.constraint = constraint
        self.label = label

    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:
        """Loops through the spans annotated by the other source, and, for each, checks
        whether they satisfy the provided constraint. If yes, adds the labelled span
        to the annotations for this source. """

        if self.other_name not in doc.spans:
            return

        for span in doc.spans[self.other_name]:
            if self.constraint(span):
                yield span.start, span.end, (self.label or span.label_)


class SpanEditorAnnotator(SpanAnnotator):
    """Annotation by editing/correcting text spans from another source 
    based on a simple editing function"""

    def __init__(self, name: str, other_name: str, editor: Callable[[Span], Span],
                 label: Optional[str] = None):
        """Creates a new annotator that looks at the annotations from the
        other_name source, and edits the span according to a given function.
        If label is other than None, the method simply reuses the same label 
        as the one specified by other_name."""

        super(SpanEditorAnnotator, self).__init__(name)
        self.other_name = other_name
        self.editor = editor
        self.label = label

    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:
        """Loops through the spans annotated by the other source and runs the
        editor function on it. """

        if self.other_name not in doc.spans:
            return

        for span in doc.spans[self.other_name]:
            edited = self.editor(span)
            if edited is not None and edited.end > edited.start:
                yield edited.start, edited.end, (self.label or span.label_)


####################################################################
# Other labelling sources
####################################################################

class VicinityAnnotator(SpanAnnotator):
    """Annotator based on cue words located in the vicinity (window of 
    surrounding words) of a given span. """

    def __init__(self, name: str, cue_words: Dict[str, str], other_name: str,
                 max_window: int = 8):
        """Creates a new annotator based on a set of cue words (each mapped 
        to a given output label) along with the name of another labelling
        source from which span candidates will be extracted."""

        super(VicinityAnnotator, self).__init__(name)

        self.cue_words = cue_words
        self.other_name = other_name
        self.max_window = max_window

    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:
        """Searches for spans that have a cue word in their vicinity - and if 
        yes, tag the span with the label associated with the cue word."""

        if self.other_name not in doc.spans:
            return
        
        # We loop on the span candidates from the other labelling source
        for span in doc.spans[self.other_name]:

            # Determine the boundaries of the context (based on the window)
            # NB: we do not wish to cross sentence boundaries
            left_bound = max(span.sent.start, span.start - self.max_window//2+1)
            right_bound = min(span.sent.end, span.end+self.max_window//2+1)

            for tok in doc[left_bound:right_bound]:
                for tok_form in [tok.text, tok.lower_, tok.lemma_]:
                    if tok_form in self.cue_words:
                        yield span.start, span.end, self.cue_words[tok_form]
