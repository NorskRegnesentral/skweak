from collections import defaultdict
from skweak.base import CombinedAnnotator
from typing import Dict, List, Optional
from spacy.tokens import Doc

class LFAnalysis:
    """ Run analyses on a list of spaCy Documents to which LFs have 
        been applied.
        
        Analyses include:
            1. Label Conflicts 
            2. Label Coverage
            3. Label Overlap
            4. Label Accuracies (against labeled dataset)
    """

    def __init__(
        self,
        docs:List[Doc],
        combined_annotator:CombinedAnnotator,
        excluded_lf_names:Optional[List[str]] = None,
        labels_to_gold_span_names:Optional[Dict[str, str]] = None
    ):
        """ Initializes LFAnalysis tool on a list of spaCY documents to which
            the CombinedAnnotator has been applied.
            
            If there are annotators that should be excluded from analysis
            these can be optionally provided.

            If `docs` have gold labels (e.g., manually assigned annotations),
            these can be optionally utilized downstream to normalize coverage
            or compute empirical accuracies of individual learning functions. 
            Gold labels should be included among the spans for the Doc objects
            and `gold_labels` should contain a mapping  between label name
            and span name.

            For example, if the following value as provided:
                labels_to_gold_span_names = {
                    'PER': 'PER_GOLD',
                    'LOC': 'LOC_GOLD
                }  

            This would imply that each Doc in `docs` will have spans named
            'PER_GOLD' and 'LOC_GOLD', with strings that have been have gold
            labels 'PER' and 'LOC' respectively.
        """
        self.docs = docs
        self.labels_to_gold_span_names = labels_to_gold_span_names
        self.lf_names = set()
        self.lf_labels = set()
        self.lf_labels_to_lf_names = defaultdict(list)

        # Index LFs from CombinedAnnotator except those explicitly excluded
        for lf in combined_annotator.annotators:
            if excluded_lf_names is not None and lf.name in excluded_lf_names:
                continue
            else:
                self._validate_span_across_docs(lf.name)
                self.lf_names.add(lf.name)
                self.lf_labels_to_lf_names[lf.label].append(lf.name)

        if self.labels_to_gold_span_names is not None:
            # Check that each gold label is addressed by 1+ LF
            missing_labels = (
                set(self.labels_to_gold_span_names.keys()) - self.lf_labels
            )
            if len(missing_labels) >= 1:
                raise ValueError(
                    f"Missing labels from LFs for: {missing_labels}"
                )

            # Check that each LF is addressed by a gold label
            missing_gold_labels = (
                self.lf_labels- set(self.labels_to_gold_span_names.keys())
            )
            if len(missing_gold_labels) >= 1:
                raise ValueError(
                    f"Missing gold labels for: {missing_gold_labels}"
                )

            # Validate that gold labels (if provided) exist for each document
            for span_name in self.labels_to_gold_span_names.values():
                self._validate_span_across_docs(span_name)
            

    def label_coverage(
        self,
        labels:Optional[List[str]] = None,
        normalize_by_gold_labels = False
    ) -> Dict[str, float]:
        """ Compute the fraction of spaCy documents that have at least one span
            with a specific label.

            If labels are provided, coverage is computed for each provided 
            label. Otherwise, label coverage is computed for all labels
            identified during initialization.

            If `normalize_by_gold_labels` is True, compute coverage for a label
            relative to the number of documents with the label as a gold label.

            For example, if 5 out of 10 documents have "PER" spans and 
            `normalize_by_gold_labels` is False, coverage will be 50%. However,
            if `normalize_by_gold_labels` is True and 5 out of 10 documents 
            have "PER" spans according to the provided gold labels, the 
            coverag will be 100%.
        """
        if labels is None:
            # Check coverage for all labels
            labels = self.lf_labels
        else:
            # Validate that label is returned by at least one LF
            for label in labels:
                self._validate_label_across_docs(label)
        
        label_counts = defaultdict(0)
        if normalize_by_gold_labels:
            gold_label_counts = defaultdict(0)

        # Count documents containing each of the labels (according to LFs)
        for doc in self.docs:
            for label in labels:
                for lf_name in self.lf_labels_to_lf_names[label]:
                    if len(doc.spans[lf_name]) >= 1:
                        label_counts[label] += 1
                        break
                if normalize_by_gold_labels and len(
                        doc.spans[self.labels_to_gold_span_names[label]]
                    ) >= 1:
                        gold_label_counts[label] += 1
                    

        # Compute coverages:
        if normalize_by_gold_labels:
            return {
                label: counts / gold_label_counts[label] 
                for label, counts in label_counts.items()
            }            
        else:
            return {
                label: counts / len(self.docs) 
                for label, counts in label_counts.items()
            }


    def _validate_span_across_docs(self, name:str):
        for d in self.docs:
            if name not in d.spans.keys():
                raise ValueError(
                    f"{name} span is missing from one"
                    "or more your the provided documents"
                )


    def _validate_label_across_docs(self, label:str):
        if label not in self.lf_labels:
            raise ValueError(
                f"{label} is not captured by your LFs"
                f"{self.lf_labels} are captured."
            )
    