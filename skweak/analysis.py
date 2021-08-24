from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas
from scipy import sparse
import scipy
from spacy.tokens import Doc

from skweak import utils


class LFAnalysis:
    """ Run analyses on a list of spaCy Documents (corpus) to which LFs have
    been applied. Analyses are conducted at a token level.
    """

    def __init__(
        self,
        corpus:List[Doc],
        labels:List[str],
        sources:Optional[List[str]] = None,
        strict_match:bool = False,
    ):
        """ Initializes LFAnalysis tool on a list of spaCY documents to which
        the sources (LFs) have been applied.

        If sources are provided, this subset of sources shall be used
        in the LF Analysis. Otherwise, the union of all sources
        (across documents) are used.

        If `strict_match` is True, labels such as I-DATE and B-DATE shall be 
        considered unique and different labels. If `strict_match` is False,
        labels such as I-DATE and B-DATE will be normalized to a single 
        label DATE. Note `strict_match` should only be set as True, when
        using labels with BIOLU format. 
        """
        self.corpus = corpus
        self.sources = self._get_corpus_sources(sources)
        (
            self.labels,
            self.label2idx,
            self.prefixes,
            self.labels_without_prefix
        ) = self._get_token_level_labels(labels, strict_match)
        self.L = self._corpus_to_token_array(strict_match)
        self._L_sparse = sparse.csr_matrix(self.L)
        self.label_row_indices = self._get_row_indices_with_labels()


    def label_conflict(self) -> pandas.DataFrame:
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

        Conflicts are computed for labels that have 1+ instances in the corpus.
        """
        result = {}
        conflicts = self._conflicted_data_points()
        for label_idx, indices in enumerate(self.label_row_indices):
            label = self.labels[label_idx]
            if label == 'O' or len(indices) == 0:
                continue
            result[label] = (np.sum(conflicts[indices]) / len(indices))
        return pandas.DataFrame.from_dict(
            result, orient='index', columns=['conflict']
        )


    def label_overlap(self) -> pandas.DataFrame:
        """ For each label, compute the fraction of tokens with at least 2 
        LFs providing a non-null annotation. Overlap computed for labels 
        that have 1+ instances in the corpus.
        """
        result = {}
        overlaps = self._overlapped_data_points()
        for label_idx, indices in enumerate(self.label_row_indices):
            label = self.labels[label_idx]
            if label == 'O' or len(indices) == 0:
                continue
            result[label] = (np.sum(overlaps[indices]) / len(indices))
        return pandas.DataFrame.from_dict(
            result, orient='index', columns=['overlap']
        )


    def lf_coverages(self) -> pandas.DataFrame:
        """ For each LF and its target labels (i.e. labels that it
        assigns 1+ times across the corpus of Docs), compute:

            # of tokens labeled by LF X with label Y
            -----------------------------------------
            # of distinct tokens labeled with label Y across all LFs
        """
        result = {}
        for label_idx, indices in enumerate(self.label_row_indices):
            label = self.labels[label_idx]
            if label == 'O' or len(indices) == 0:
                continue
            covered = self._covered_by_label(label_idx)
            result[label] = covered / len(indices)
        return pandas.DataFrame.from_dict(
            result, orient='index', columns=self.sources)


    def lf_conflicts(self) -> pandas.DataFrame:
        """ For each LF, compute the fraction of tokens that have conflicting
        non-null annotations overall and for each of its target labels.

        A conflict is defined as an instance where 2 LFs with different
        target labels annotate the same token with a non-null label. LF
        conflicts computed for labels that have 1+ instance in the corpus
        from the given LF.
        """
        return pandas.DataFrame()


    def lf_overlaps(self) -> pandas.DataFrame:
        """ For each LF and its target labels, compute the fraction of tokens
        that have another LF providing a non-null label.
        """
        return pandas.DataFrame()


    def lf_empirical_accuracies(self,
        Y:List[Doc],
        label_to_span_names:Dict[str, str]
    ) -> pandas.DataFrame:
        """ For each LF and its target labels, compute the empirical accuracy.
        """
        return pandas.DataFrame()


    # ----------------------
    # Initialization Helpers
    # ----------------------
    def _get_token_level_labels(
        self,
        original_labels:List[str],
        strict_match: bool
    ) -> Tuple[List[str], Dict[str, int], Set[str], Set[str]]:
        """ Generate helper dictionaries that normalize and index labels
        used for token-level analyses.
        """
        # 0-th label should be 'O' (null token) for token level analyses
        if 'O' not in original_labels:
            original_labels.insert(0, 'O')
        elif original_labels[0] != 'O':
            original_labels.remove('O')
            original_labels.insert(0, 'O')
        
        # Normalize labels if strict matching is disabled (e.g., convert
        # I-PER and B-PER to PER). Also construct mapping of label namess
        # to indices, identify prefixes in original label set, and 
        # labels without prefixes according to original label set. 
        label2idx, prefixes, labels_without_prefix = utils._index_labels(
            original_labels,
            not strict_match
        )

        # Generate mapping of index to label name
        labels = [label for label in label2idx.keys()]
        return labels, label2idx, prefixes, labels_without_prefix


    def _get_corpus_sources(self, sources:Optional[List[str]]) -> List[str]:
        """ Determine sources for analysis. If no sources are provided, sources
        is computed as the union of sources used across the corpus of Docs.
        """
        corpus_sources = set()
        if sources is None:
            for doc in self.corpus:
                corpus_sources.update(set(doc.spans.keys()))
            return list(corpus_sources)
        else:
            return list(set(sources))


    def _corpus_to_token_array(self, strict_match:bool) -> np.ndarray:
        """ Convert corpus to a matrix of dimensions:
        (# of tokens in corpus, # sources)
        """
        return np.concatenate([
            utils._spans_to_array(
                doc,
                self.sources,
                self.label2idx,
                self.labels_without_prefix,
                self.prefixes if strict_match else None
            ) for doc in self.corpus
        ])

    # ----------------
    # Analysis Helpers
    # ----------------
    def _conflicted_data_points(self) -> np.ndarray:
        """Get indicator vector z where z_i = 1 if x_i is labeled differently
        by two LFs."""
        m = sparse.diags(np.ravel(self._L_sparse.max(axis=1).todense()))
        return np.ravel(
            np.max(m @ (self._L_sparse != 0) != self._L_sparse, axis=1)
            .astype(int)
            .todense()
        )


    def _covered_data_points(self) -> np.ndarray:
        """Get indicator vector z where z_i = 1 if x_i is
        labeled by at least one LF."""
        return np.ravel(np.where(self._L_sparse.sum(axis=1) != 0, 1, 0))


    def _covered_by_label(self, label_val:int) -> np.ndarray:
        """Get count vector c where c_i is the # of times the ith source
        predicted a token to have the label value"""
        return np.ravel((self._L_sparse == label_val).sum(axis=0))


    def _overlapped_data_points(self) -> np.ndarray:
        """Get indicator vector z where z_i = 1 if x_i i
        labeled by more than one LF."""
        return np.where(np.ravel((self._L_sparse != 0).sum(axis=1)) > 1, 1, 0)


    def _infer_target_labels(self) -> np.ndarray:
        """Infer the target labels each of the sources, by examining the 
        labels in the corpus of Documents.
        """
        return np.unique(self.L, axis=0)


    def _get_row_indices_with_labels(self) -> List[int]:
        """ Determine which rows have been assigned a given label by at least
        1 label functions.
        """
        cols = np.arange(self.L.size)
        m = sparse.csr_matrix((cols, (self.L.ravel(), cols)),
                        shape=(self.L.max() + 1, self.L.size))
        return [
            np.unique(np.unravel_index(row.data, self.L.shape)[0])
            for row in m
        ]