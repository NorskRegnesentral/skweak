
from skweak.gazetteers import GazetteerAnnotator
from skweak import utils
from typing import Dict, List, Tuple, Iterable
from . import gazetteers, base
from spacy.tokens import Doc, Span  # type: ignore
from collections import defaultdict


class DocumentHistoryAnnotator(base.SpanAnnotator):
    """Annotation based on the document history: 
    1) if a person name has been mentioned in full (at least two consecutive tokens, 
    most often first name followed by last name), then mark future occurrences of the 
    last token (last name) as a PER as well. 
    2) if an organisation has been mentioned together with a legal type, mark all other 
    occurrences (possibly without the legal type at the end) also as a COMPANY.
    """

    def __init__(self, basename: str, other_name: str, labels: List[str],
                 case_sentitive=True):
        """Creates a new annotator looking at the global document context, based on another 
        annotation layer (typically a layer aggregating existing annotations). Only the 
        labels specified in the argument will be taken into account."""

        super(DocumentHistoryAnnotator, self).__init__(basename)
        self.other_name = other_name
        self.labels = labels
        self.case_sensitive = case_sentitive

    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:
        """Search for spans on one single document"""

        # Extract the first mentions of each entity
        first_observed = self.get_first_mentions(doc)

        # We construct tries based on the first mentions
        tries = {label: gazetteers.Trie() for label in self.labels}
        first_observed_bounds = set()
        for tokens, span in first_observed.items():
            tries[span.label_].add(tokens)
            first_observed_bounds.add((span.start, span.end))

        gazetteer = GazetteerAnnotator(self.name, tries, case_sensitive=self.case_sensitive,
                                       additional_checks=not self.case_sensitive)

        for start, end, label in gazetteer.find_spans(doc):
            if (start, end) not in first_observed_bounds:
                yield start, end, label

        return doc

    def get_first_mentions(self, doc) -> Dict[List[str], Span]:
        """Returns a set containing the first mentions of each entity as triples
        (start, end, label) according to the "other_name' layer.

        The first mentions also contains subsequences: for instance, a named entity
        "Pierre Lison" will also contain the first mentions of ['Pierre'] and ['Lison'].
        """
        if self.other_name not in doc.spans:
            return {}

        first_observed = {}
        for span in doc.spans[self.other_name]:

            # NB: We only consider entities with at least two tokens
            if span.label_ not in self.labels or len(span) < 2:
                continue

            # We also extract subsequences
            for length in range(1, len(span)+1):
                for i in range(length, len(span)+1):

                    start2 = span.start + i-length
                    end2 = span.start + i
                    subseq = tuple(tok.text for tok in doc[start2:end2])

                    # We ony consider first mentions
                    if subseq in first_observed:
                        continue

                    # To avoid too many FPs, the mention must have at least 4 charactes
                    if sum(len(tok) for tok in subseq) <4:
                        continue
                    
                    # And if the span looks like a proper name, then at least one 
                    # token in the subsequence must look like a proper name too 
                    if (any(utils.is_likely_proper(tok) for tok in span) and not 
                          any(utils.is_likely_proper(tok) for tok in doc[start2:end2])):
                        continue
                        
                    first_observed[subseq] = Span(doc, start2, end2, span.label_)

        return first_observed


class DocumentMajorityAnnotator(base.SpanAnnotator):
    """Annotation based on majority label for the same entity string elsewhere in the 
    document. The annotation creates two layers for each label, one for case-sensitive 
    occurrences of the entity string in the document, and one for case-insensitive 
    occurrences.
    """

    def __init__(self, basename: str, other_name: str, case_sensitive=True):
        """Creates a new annotator that looks at (often aggregated) annotations from
        another layer, and annotates entities based on their majority label elsewhere
        in the document. """

        super(DocumentMajorityAnnotator, self).__init__(basename)
        self.other_name = other_name
        self.case_sensitive = case_sensitive

    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:
        """Generates span annotations for one single document based on 
        majority labels"""

        # We search for the majority label for each entity string
        majority_labels = self.get_majority_labels(doc)

        # we build trie to easily search for these entities in the text
        tries = {label: gazetteers.Trie()
                 for label in set(majority_labels.values())}
        for ent_tokens, label in majority_labels.items():
            tries[label].add(list(ent_tokens))

        gazetteer = GazetteerAnnotator(self.name, tries, self.case_sensitive,
                                       additional_checks=not self.case_sensitive)
        for start, end, label in gazetteer.find_spans(doc):
            yield start, end, label

    def get_majority_labels(self, doc: Doc) -> Dict[Tuple[str], str]:
        """Given a document, searches for the majority label for each entity string
        with at least self.min_counts number of occurrences. """

        # Get the counts for each label per entity string
        # (and also for each form, to take various casings into account)
        label_counts = defaultdict(dict)
        form_counts = defaultdict(dict)
        spans = utils.get_spans_with_probs(doc, self.other_name)

        all_tokens_low = [tok.lower_ for tok in doc]
        checked = {}
        for span, prob in spans:

            # We only apply document majority for strings occurring more than once
            tokens_low = tuple(all_tokens_low[span.start:span.end])
            if tokens_low not in checked:
                occurs_several_times = utils.at_least_nb_occurrences(
                    tokens_low, all_tokens_low, 2)
                checked[tokens_low] = occurs_several_times
            else:
                occurs_several_times = checked[tokens_low]

            # If the string occurs more than once, update the counts
            if occurs_several_times:

                label_counts[tokens_low][span.label_] = label_counts[tokens_low].get(
                    span.label_, 0) + prob
                tokens = tuple(tok.text for tok in span)
                form_counts[tokens_low][tokens] = form_counts[tokens_low].get(
                    tokens, 0) + prob

        # Search for the most common label for each entity string
        majority_labels = {}
        for lower_tokens, labels_for_ent in label_counts.items():
            majority_label = max(
                labels_for_ent, key=lambda x: labels_for_ent[x])
            forms_for_ent = form_counts[lower_tokens]
            majority_form = max(forms_for_ent, key=lambda x: forms_for_ent[x])

            majority_labels[majority_form] = majority_label

        return majority_labels
