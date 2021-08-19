from collections import defaultdict
from skweak.base import CombinedAnnotator
from typing import DefaultDict, Dict, List, Optional, Set
from spacy.tokens import Doc

class LFAnalysis:
    """ Run analyses on a list of spaCy Documents (corpus) to which LFs have
    been applied.
    """

    def __init__(
        self,
        corpus:List[Doc],
        labels:List[str],
        combined_annotator:CombinedAnnotator,
        excluded_lf_names:Optional[List[str]] = None,
    ):
        """ Initializes LFAnalysis tool on a list of spaCY documents to which
        the CombinedAnnotator has been applied.
        
        If there are annotators that should be excluded from analysis
        these can be optionally provided.

        If there are labels for which no spans have been annotated an
        error message shall be printed, but no Exception shall be thrown.
        """
        self.corpus = corpus
        self.labels = labels
        self.lf_names = self._index_lfs(combined_annotator,excluded_lf_names)
        self.labels_to_lf_names = self._map_labels_to_lfs()


    def label_conflict(self) -> Dict[str, float]:
        """ For each label, compute the fraction of tokens with conflicting
        non-null labels. 
        
        A conflict is defined as an instance where 2 LFs that annotate 
        the same token with different non-null labels. As an example, a
        conflict would be detected if:
            - LF1 returns "PER" for the token "Apple"
            - LF2 returns "ORG" for the token "Apple"

        A conflict is not registered if 1 LF predicts the token to have
        a null label, while another predicts the token to have non-null
        label. For example, a conflict would not be registered if: 
            - LF1 returns "ORG" for the token "Apple"
            - LF2 returns "O" (null-label) for the token "Apple"
        """
        return {}


    def label_agreement(self) -> Dict[str, float]:
        """ For each label, compute the fraction of tokens with agreeing
        non-null labels. 
        
        An agreement is defined as an instance where 2 LFs that annotate 
        the same token with the same non-null labels. As an example, an
        agreement would be detected if:
            - LF1 returns "ORG" for the token "Apple"
            - LF2 returns "ORG" for the token "Apple"
        """
        return {}


    def label_overlap(self) -> Dict[str, float]:
        """ For each label, compute the fraction of tokens with at least 2 
        LFs providing a non-null annotation.
        """
        return {}


    def lf_conflicts(
        self
    )-> Dict[str, Dict[str, float]]:
        """ For each LF, compute the fraction of tokens that have conflicting
        non-null annotations overall and for each of its target labels.

        A conflict is defined as an instance where 2 LFs with different
        target labels annotate the same token with a non-null label.

        Example return object:
        {
            "lf1": {
                "PER": 0.4,
                "DATE" 0.0,
            }, 
            "lf2": {
                "PER": 0.2,
            }
        }

        """
        return {}


    def lf_agreements(self) -> Dict[str, Dict[str, float]]:
        """ For each LF and its target labels, compute the fraction of tokens
        that have agreeing non-null annotations from another LF for each 
        of its target labels.

        An agreement is defined as an instance where 2 LFs with the same
        target labels annotate the same token with a non-null label.

        Example return object:
        {
            "lf1": {
                "PER": 0.6,
                "DATE" 1.0,
            }, 
            "lf2: {
                "PER": 0.2,
            }
        }
        """
        return {}


    def lf_overlaps(self) -> Dict[str, Dict[str, float]]:
        """ For each LF and its target labels, compute the fraction of tokens
        that have another LF providings a non-null label.

        Example return object:
        {
            "lf1": {
                "PER": 0.6,
                "DATE" 1.0,
            }, 
            "lf2": {
                "PER": 0.2,
            }
        }
        """
        return {}


    def lf_empirical_accuracies(self,
        Y:List[Doc],
        label_to_span_names:Dict[str, str]
    ) -> Dict[str, Dict[str, float]]:
        """ For each LF and its target labels, compute the empirical accuracy.

        Example return object:
        {
            "lf1": {
                "PER": 0.6,
                "DATE" 1.0,
            }, 
            "lf2": {
                "PER": 0.2,
            }
        }
        """
        return {}


    def lf_labels(self) -> DefaultDict[Set[str]]:
        """ Infer the labels of each LF based on the evidence in the corpus.
        """
        lf_names_to_labels = defaultdict(set)
        for label, lf_names in self.labels_to_lf_names.items():
            for lf_name in lf_names:
                lf_names[lf_name].add(label)
        return lf_names_to_labels


    def _index_lfs(
        self,
        combined_annotator:CombinedAnnotator,
        excluded_lf_names:Optional[List[str]] = None
    ):
        """ Index LFs from CombinedAnnotator except those explicitly excluded.
        """
        self.lf_names:Set[str] = set()
        for lf in combined_annotator.annotators:
            if excluded_lf_names is not None and lf.name in excluded_lf_names:
                continue
            else:
                self._validate_lf_across_corpus(lf.name)
                self.lf_names.add(lf.name)


    def _map_labels_to_lfs(self):
        """ Generate mapping from labels to LFs, given corpus of docs annotated
            by LFs. Raise a warning if there is a label selected for analysis
            that has never been seen in the annotated dataset. 
        """
        self.labels_to_lf_names = defaultdict(set)
        for doc in self.corpus:
            for lf_name in self.lf_names:
                for span in doc.spans[lf_name]:
                    self.labels_to_lf_names[span.label_].add(lf_name)
        
        unused_labels = []
        for label in self.labels:
            if len(self.labels_to_lf_names) == 0:
                unused_labels.append(label)
        print(
            f"{unused_labels} labels were not found"
            "in your corpus of documents"
        )

    def _validate_lf_across_corpus(self, name:str):
        """ Check whether a LF has been applied to every document in a
            corpus. 
        """
        for d in self.corpus:
            if name not in d.spans.keys():
                raise ValueError(
                    f"{name} LF is missing from one"
                    "or more of documents in the corpus"
                )
