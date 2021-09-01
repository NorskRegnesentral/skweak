from collections import defaultdict
from operator import imod
from typing import DefaultDict, Dict, List, Optional, Tuple, Set

import numpy as np
from numpy.lib.type_check import nan_to_num
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

        If `strict_match` is False,labels such as I-DATE and B-DATE will be
        normalized to a single label DATE. 
        
        Note `strict_match` should only be set as True, when using labels
        with BIOLU format. If `strict_match` is True, labels such as I-DATE
        and B-DATE shall be considered unique and different labels.
        """
        self.corpus = corpus
        self.sources, self.sources2idx = self._get_corpus_sources(sources)
        self.strict_match = strict_match
        (
            self.labels,
            self.label2idx,
            self.prefixes,
            self.labels_without_prefix
        ) = self._get_token_level_labels(labels)
        self.L = self._corpus_to_token_array(self.corpus, self.sources)
        self._L_sparse = sparse.csr_matrix(self.L)
        self.label_row_indices = self._get_row_indices_with_labels()


    def label_overlap(self, nan_to_num:float = 0.) -> pandas.DataFrame:
        """ For each label, compute the fraction of tokens with at least 2 
        LFs providing a non-null annotation. 
        """
        result = {}
        overlaps = self._overlapped_data_points()
        for label_idx, indices in enumerate(self.label_row_indices):
            label = self.labels[label_idx]
            if label == 'O':
                continue
            if len(indices) == 0:
                result[label] = nan_to_num
            else:
                result[label] = np.sum(overlaps[indices]) / len(indices)
        return pandas.DataFrame.from_dict(
            result, orient='index', columns=['overlap']
        )


    def label_conflict(self, nan_to_num:float = 0.) -> pandas.DataFrame:
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
        result = {}
        conflicts = self._conflicted_data_points()
        for label_idx, indices in enumerate(self.label_row_indices):
            label = self.labels[label_idx]
            if label == 'O':
                continue
            if len(indices) == 0:
                result[label] = nan_to_num
            else:
                result[label] = np.sum(conflicts[indices]) / len(indices)
        return pandas.DataFrame.from_dict(
            result, orient='index', columns=['conflict']
        )


    def lf_target_labels(self) -> Dict[str, List[int]]:
        """Infer the target labels of each LF based on evidence in the
        label matrix. Excludes null token label.

        NB: If `strict_match` is true for the LFAnalysis instance the target
        labels will be comprised of BILU labels for each normalized target
        label in a LF's domain. For example, if a LF returns [NULL,  I-PERSON,
        B-PERSON, L-PERSON] across the samples in the corpus, the target set
        will still be [NULL,  I-PERSON, B-PERSON, L-PERSON, U-PERSON].
        """
        if self.strict_match:
            target_label_idxs = {}
            for i in range(self._L_sparse.shape[1]):
                target_label_idxs_with_prefixes = set()
                for label_idx in set(self._L_sparse[:, i].data):
                    label_without_prefix = self.labels[label_idx].split('-')[1]
                    for prefix in 'BILU':
                        target_label_idxs_with_prefixes.add(
                            self.label2idx[
                                '{}-{}'.format(prefix, label_without_prefix)
                            ]
                        )
                target_label_idxs[self.sources[i]] = list(
                    target_label_idxs_with_prefixes
                )
        else:
            target_label_idxs = {
                self.sources[i]: sorted(list(set(self._L_sparse[:, i].data)))
                for i in range(self._L_sparse.shape[1])
            }
        return target_label_idxs


    def lf_coverages(
        self,
        agg:bool = False,
        nan_to_num:float = 0.
    ) -> pandas.DataFrame:
        """ Compute LF coverages (i.e., tokens labeled by a LF that 
        are also labeled by another LF).

        If `agg` is True, coverages are computed for each LF across all of the
        LF's target labels:

        Coverage (LF X) = 
            # of tokens labeled non-null by LF X
            -------------------------------------------------------------------
            # of tokens labeled non-null by all LFs across LF X's target labels
        
        If `agg` is False, coverages are computed individually for each target
        label and LF:

        Coverage (LF X, Label Y) = 
            # of tokens labeled by LF X as Y
            --------------------------------------------------------
            # of distinct tokens labeled as Y across all LFs

        NB: If `strict_match` is true for the LFAnalysis instance the target
        labels will be comprised of BILU labels for each normalized target
        label in a LF's domain. For example, if a LF returns [NULL,  I-PERSON,
        B-PERSON, L-PERSON] across the samples in the corpus, the target set
        will still be [NULL,  I-PERSON, B-PERSON, L-PERSON, U-PERSON].
        """
        if agg:
            # Compute the number of tokens covered by each LF
            covered_token_counts = np.ravel((self._L_sparse != 0).sum(axis=0))

            # Compute number of tokens covered by target labels for a LF
            total_token_counts = np.zeros(len(self.sources))
            for lf, lf_target_labels in self.lf_target_labels().items():
                label_coverages = np.zeros((self.L.shape[0], 1))
                lf_idx = self.sources2idx[lf]
                for label_idx in lf_target_labels:
                    label_coverages += self._covered_by_label(label_idx)
                union_label_coverages = label_coverages >= 1
                total_token_counts[lf_idx] = np.sum(union_label_coverages)
            
            # Compute LF Coverages
            coverages = np.nan_to_num(
                covered_token_counts/total_token_counts,
                nan=nan_to_num
            )
            return pandas.DataFrame(
                coverages.reshape(1, len(self.sources)),
                columns=self.sources,
            )                        
        else:
            result = {}
            for label_idx, indices in enumerate(self.label_row_indices):
                label = self.labels[label_idx]
                if label == 'O':
                    continue
                covered = self._covered_by_label_counts(label_idx)
                if len(indices) == 0:
                    result[label] = (
                        np.ones(len(self.sources)) * nan_to_num
                    )
                else:
                    result[label] = covered / len(indices)
            return pandas.DataFrame.from_dict(
                result, orient='index', columns=self.sources)


    def lf_overlaps(
        self,
        agg:bool = False,
        nan_to_num:float = 0.
    ) -> pandas.DataFrame:
        """ Compute LF overlaps (i.e., tokens labeled by 2+ LFs).

        If `agg` is True, overlaps are computed for each LF across all of the
        LF's target labels:

        Overlaps(LF X)= 
            # of tokens labeled non-null by LF X and another LF
            ---------------------------------------------------
            # of tokens labeled non-null by LF X

        If `agg` is False, overlaps are computed individually for each target
        label and LF:

        Overlaps(LF X, Label Y) = 
            # of tokens labeled by LF X as Y and labeled non-null by another LF
            -------------------------------------------------------------------
            # of tokens labeled by LF X as Y
        """
        if agg:
            L_sparse_indicator = (self._L_sparse != 0)
            overlaps = np.nan_to_num(
                L_sparse_indicator.T
                @ self._overlapped_data_points()
                / L_sparse_indicator.sum(axis=0),
                nan=nan_to_num
            )
            return pandas.DataFrame(
                overlaps,
                columns=self.sources,
            )
        else:
            result = {}
            overlaps = self._overlapped_data_points()
            for label_idx, indices in enumerate(self.label_row_indices):
                label = self.labels[label_idx]
                if label == 'O':
                    continue
                with np.errstate(divide='ignore',invalid='ignore'):
                    # Select rows that contain the given label and then create
                    # and indicator matrix for the label (e.g., 1 if 
                    # label was applied by LF)
                    x = (self._L_sparse[indices] == label_idx)

                    # For each LF identify the number of times the label has
                    # been selected, during a overlap and divide by the 
                    # number of times that the label was assigned y the 
                    # given label function
                    lf_overlaps = (x.T @ overlaps[indices]).T / x.sum(axis=0)
                    result[label] = np.ravel(
                        np.nan_to_num(
                            lf_overlaps,
                            nan=nan_to_num
                    ))
            
            return pandas.DataFrame.from_dict(
                result, orient='index', columns=self.sources)


    def lf_conflicts(
        self,
        agg:bool = False,
        nan_to_num:float = 0.
    ) -> pandas.DataFrame:
        """ Compute LF conflicts (i.e., instances where 2 LFs assign different
        non-null labels to a token).
        
        If `agg` is True, conflicts are computed for each LF across all of the
        LF's target labels.

        Conflicts(LF X)= 
            # of tokens labeled non-null by LF X w/ conflicting overlaps
            ------------------------------------------------------------
            # of tokens labeled non-null by LF X

        If `agg` is False, overlaps are computed individually for each target
        label and LF:

        Conflicts(LF X, Label Y) = 
            # of tokens labeled by LF X as Y w/ conflicting overlaps
            --------------------------------------------------------
            # of tokens labeled by LF X as Y
        
        """
        if agg:
            L_sparse_indicator = (self._L_sparse != 0)
            conflicts = np.nan_to_num(
                L_sparse_indicator.T
                @ self._conflicted_data_points()
                / L_sparse_indicator.sum(axis=0),
                nan=nan_to_num
            )
            return pandas.DataFrame(
                conflicts,
                columns=self.sources,
            )
        else:
            result = {}
            conflicts = self._conflicted_data_points()
            for label_idx, indices in enumerate(self.label_row_indices):
                label = self.labels[label_idx]
                if label == 'O':
                    continue
                with np.errstate(divide='ignore',invalid='ignore'):
                    # Select rows that contain the given label and then create
                    # and indicator matrix for the label (e.g., 1 if 
                    # label was applied by LF)
                    x = (self._L_sparse[indices] == label_idx)

                    # For each LF identify the number of times the label has
                    # been selected, during a conflict and divide by the 
                    # number of times that the label was assigned y the 
                    # given label function
                    lf_conflicts = (x.T @ conflicts[indices]).T / x.sum(axis=0)
                    result[label] = np.ravel(
                        np.nan_to_num(
                            lf_conflicts,
                            nan=nan_to_num
                    ))
            
            return pandas.DataFrame.from_dict(
                result, orient='index', columns=self.sources)


    def lf_empirical_accuracies(
        self,
        Y:List[Doc],
        gold_span_name:str,
        gold_labels:List[str],
        agg:bool = False,
        nan_to_num:float = 0.,
        print_warnings:bool = True,
    ) -> pandas.DataFrame:
        """ Compute empirical accuracies. 

        If `agg` is True, accuracies are computed for each LF across all of the
        LF's target labels:

        Accuracy(LF X)= 
            # of tokens labeled correctly by LF X
            -------------------------------------
            # total tokens

        If `agg` is False, accuracies are computed individually for each target
        label and LF:

        Accuracy(LF X, Label Y) = 
            # of tokens labeled correctly by LF as Y or Null Token
            ------------------------------------------------------------
            # of tokens labeled as Y or Null Token in Gold Data

        NB:
        - We assume Y has the same docs as the corpus
        - Any ground truth labels that are not covered by the LF are set to 0.
                  
          For example, if LF1 has a target label set [0, 1, 2], the
          ground truth for a dataset is [1, 2, 3, 4], and `agg` is True,
          the LF1's accuracies will be computed against the ground truth
          labels [1, 2, 0, 0]).

          Similarly, if LF1 has a target label set [0, 1, 2], the
          ground truth for a dataset is [1, 2, 3, 4], and `agg` is False,
          and we are computing the accuracy for LF1 and label 2, the
          (LF1, Label 2) accuracy will be computed against the ground truth
          labels [0, 2, 0, 0]).

        - We assume that all gold labels are contained within a single span
        and that labels do not contain prefixes (e.g. PERSON is used, not
        I-PERSON, etc.).

        - If we encounter a label that has not been indexed by the LFAnalysis
        instance the token is assigned the null label (0).

        - If `strict_match` is true for the LFAnalysis instance the target
        labels will be comprised of BILU labels for each normalized target
        label in a LF's domain. For example, if a LF returns [NULL,  I-PERSON,
        B-PERSON, L-PERSON] across the samples in the corpus, the target set
        will still be [NULL,  I-PERSON, B-PERSON, L-PERSON, U-PERSON].
        """
        # Check for same number of docs
        assert (len(self.corpus) == len(Y))

        # Determine if there are labels within the LF Analysis object
        # that do not exist in the gold labels -- we'll exclude
        # these from accuracy analyses
        missing_labels = set(self.labels) - set(gold_labels)
        if len(missing_labels) and print_warnings:
            print("WARNING: \
            The following are not presented in the gold dataset: \
            {}".format(missing_labels)
            )

        # Create Y labels matrix
        Y_L_sparse = sparse.csr_matrix(
            self._corpus_to_token_array(Y, [gold_span_name])
        )

        if agg:
            result = {}
            for lf, idxs in self.lf_target_labels().items():
                lf_idx = self.sources2idx[lf]

                # Since target labels don't include 0 (null token)
                # we manually introduce the null token as a target label
                idxs.append(0)

                # Identify labels that are outside of LF's domain
                non_target_label_idxs = list(
                    set(self.label2idx.values()) - set(idxs)
                )

                # Null out the values in the ground truth vector 
                # that have labels outside the domain of the LF
                gt = Y_L_sparse.copy()
                for label_idx in non_target_label_idxs:
                    gt[self._get_indices_for_lf_with_label(
                        Y_L_sparse,
                        0, # 0th index = gold label column in gt matrix
                        label_idx
                        )] = 0

                # Determine agreement between ground truth and LF predictions
                # Sparse matrices can compute disagreement quickly (not 
                # agreement); so we shall compute disagreement and take
                # converse to compute accuracy
                disagreement = (gt != self._L_sparse.getcol(lf_idx))

                # Compute accuracy
                result[lf] = 1.0 - (disagreement.sum() / (self.L.shape[0]))

            return pandas.DataFrame.from_dict(
                result, orient='index', columns=['acc'])
        else:
            result = defaultdict(dict)
            for lf, idxs in self.lf_target_labels().items():
                lf_idx = self.sources2idx[lf]

                # Null out the values in the ground truth vector 
                # other than the given label
                for label_idx in idxs:
                    gt = (Y_L_sparse.copy() == label_idx) 
                    preds = (self._L_sparse.getcol(lf_idx) == label_idx)

                    # Sparse matrices can compute disagreement quickly (not 
                    # agreement); so we shall compute disagreement and take
                    # converse to compute accuracy
                    disagreement = (gt != preds)

                    # Compute accuracy
                    result[lf][self.labels[label_idx]] = (
                        1 - (disagreement.sum() / (self.L.shape[0]))
                    )

            return pandas.DataFrame.from_dict(
                result,
                orient='index',
                columns=self.labels[1:]
            ).fillna(nan_to_num)


    def lf_empirical_scores(
        self,
        Y:List[Doc],
        gold_span_name:str,
        gold_labels:List[str],
        agg:bool = False,
        nan_to_num:float = 0.,
        print_warnings:bool = True,
    ) -> DefaultDict(dict):
        """ Compute empirical precision, recall, and f1. 

        If `agg` is True, conflicts are computed for each LF across all of the
        LF's target labels:

        Precision(LF X)= 
            # of tokens labeled non-null and correctly by LF X 
            --------------------------------------------------
            # total tokens labeled non-null by LF X 

        Recall(LF X)= 
            # of tokens labeled non-null and correctly by LF X 
            ----------------------------------------------------------
            # total tokens assigned target labels of LF X by Gold Data 

        If `agg` is False, overlaps are computed individually for each target
        label and LF:

        Precision(LF X, Label Y) = 
            # of tokens correctly labeled as Y by LF X 
            ------------------------------------------------------------
            # total tokens labeled as by LF X 

        Recall(LF X, Label Y) = 
            # of tokens correctly labeled as Y by LF X 
            ------------------------------------------------------------
            # total tokens assigned Y by Gold Data 

        Regardless of `agg` strategy:
            F1 = (2 * Precision * Recall)
                -------------------------
                (Precision + Recall)                        
        NB:
        - We assume Y has the same docs as the corpus
        - Any ground truth labels that are not covered by the LF are set to 0.
                  
          For example, if LF1 has a target label set [0, 1, 2], the
          ground truth for a dataset is [1, 2, 3, 4], and `agg` is True,
          the LF1's accuracies will be computed against the ground truth
          labels [1, 2, 0, 0]).

          Similarly, if LF1 has a target label set [0, 1, 2], the
          ground truth for a dataset is [1, 2, 3, 4], and `agg` is False,
          and we are computing the accuracy for LF1 and label 2, the
          (LF1, Label 2) accuracy will be computed against the ground truth
          labels [0, 2, 0, 0]).

        - We assume that all gold labels are contained within a single span
        and that labels do not contain prefixes (e.g. PERSON is used, not
        I-PERSON, etc.).

        - If we encounter a label that has not been indexed by the LFAnalysis
        instance the token is assigned the null label (0).

        - If there are labels that are indexed by LFAnalysis but are not 
        included in the gold dataset, we exclude these labels from the
        analysis.

        - If `strict_match` is true for the LFAnalysis instance the target
        labels will be comprised of BILU labels for each normalized target
        label in a LF's domain. For example, if a LF returns [NULL,  I-PERSON,
        B-PERSON, L-PERSON] across the samples in the corpus, the target set
        will still be [NULL,  I-PERSON, B-PERSON, L-PERSON, U-PERSON].
        """
        # Check for same number of docs
        assert (len(self.corpus) == len(Y))

        # Determine if there are labels within the LF Analysis object
        # that do not exist in the gold labels -- we'll exclude
        # these from accuracy analyses (note this does not check
        # for missing BILU labels, e.g. checks for PERSON missing
        # but not B-PERSON missing)
        missing_labels_without_prefixes = (
            self.labels_without_prefix - set(gold_labels)
        )
        if self.strict_match:
            missing_labels = set([
                '{}-{}'.format(prefix, label) for prefix
                in 'BILU' for label in missing_labels_without_prefixes
            ])
        else:
            missing_labels = missing_labels_without_prefixes

        if len(missing_labels_without_prefixes) and print_warnings:
            print("WARNING: \
            The following are not presented in the gold dataset: \
            {}".format(missing_labels_without_prefixes)
            )

        # Create Y labels matrix
        Y_L_sparse = sparse.csr_matrix(
            self._corpus_to_token_array(Y, [gold_span_name])
        )

        if agg:
            counts = defaultdict(lambda: defaultdict(int))
            result = {}
            for lf, idxs in self.lf_target_labels().items():
                lf_idx = self.sources2idx[lf]

                for label_idx in idxs:
                    if self.labels[label_idx] in missing_labels:
                        continue

                    # Create indicator matrix for label in gt
                    gt = (Y_L_sparse.copy() == label_idx) 

                    # Find rows, cols of tokens predicted as label by LF
                    (pred_rows_idxs, pred_cols_idxs) = \
                        self._get_indices_for_lf_with_label(
                            self._L_sparse,
                            lf_idx, label_idx
                        )

                    # Find the number of predictions that were correct
                    # for the label
                    correct_count = (
                        gt[pred_rows_idxs, pred_cols_idxs] == 1
                    ).sum()

                    counts[lf]['num_correct_predicted'] += correct_count
                    counts[lf]['num_predicted'] += len(pred_cols_idxs)
                    counts[lf]['num_ground_truth'] += gt.sum()

                # Compute precision
                if counts[lf]['num_predicted'] == 0:
                    precision = nan_to_num
                else:
                    precision = (counts[lf]['num_correct_predicted'] / 
                        counts[lf]['num_predicted'])

                # Compute recall
                if counts[lf]['num_ground_truth'] == 0:
                    recall = nan_to_num
                else:
                    recall = (
                        counts[lf]['num_correct_predicted'] /
                        counts[lf]['num_ground_truth']
                    )

                # Compute F1
                if (counts[lf]['num_ground_truth'] == 0 or 
                    counts[lf]['num_correct_predicted'] == 0):
                    f1 = nan_to_num
                else:
                    if precision + recall == 0.0:
                        f1 = 0.0
                    else:
                        f1 = (2 * precision * recall) / (precision + recall)

                result[lf] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                }
            return result

        else:
            result = defaultdict(dict)
            for lf, idxs in self.lf_target_labels().items():
                lf_idx = self.sources2idx[lf]

                for label_idx in idxs:
                    if self.labels[label_idx] in missing_labels:
                        continue

                    # Create indicator matrix for label in gt
                    gt = (Y_L_sparse.copy() == label_idx) 

                    # Find rows, cols of tokens predicted as label by LF
                    (pred_rows_idxs, pred_cols_idxs) = \
                        self._get_indices_for_lf_with_label(
                            self._L_sparse,
                            lf_idx, label_idx
                        )

                    # Find the number of predictions that were correct
                    # for the label
                    correct_count = (
                        gt[pred_rows_idxs, pred_cols_idxs] == 1
                    ).sum()

                    # Compute precision
                    if len(pred_cols_idxs) == 0:
                        precision = nan_to_num
                    else:
                        precision = correct_count / len(pred_cols_idxs)
                    
                    # Compute recall
                    num_gt = gt.sum()
                    if num_gt == 0:
                        recall = nan_to_num
                    else:
                        recall = correct_count / num_gt

                    # Compute f1
                    if num_gt == 0 or len(pred_cols_idxs) == 0:
                        f1 = nan_to_num
                    else:
                        if precision + recall == 0.0:
                            f1 = 0.0
                        else:
                            f1 = (2 * precision * recall) / (precision + recall)

                    result[lf][self.labels[label_idx]] = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                    }

            return result

    # ----------------------
    # Initialization Helpers
    # ----------------------
    def _get_token_level_labels(
        self,
        original_labels:List[str]
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
            not self.strict_match
        )

        # Generate mapping of index to label name
        labels = [label for label in label2idx.keys()]
        return labels, label2idx, prefixes, labels_without_prefix


    def _get_corpus_sources(
        self,
        sources:Optional[List[str]]
    ) -> Tuple[List[str], Dict[str, int]]:
        """ Determine sources for analysis. If no sources are provided, sources
        is computed as the union of sources used across the corpus of Docs.
        """
        result_sources = []
        corpus_sources = set()
        if sources is None:
            for doc in self.corpus:
                corpus_sources.update(set(doc.spans.keys()))
            result_sources = list(corpus_sources)
        else:
            result_sources = list(set(sources))
        return result_sources, {
            source: i for i, source in enumerate(result_sources)
        }


    def _corpus_to_token_array(
        self,
        corpus:List[Doc],
        sources:List[str]
    ) -> np.ndarray:
        """ Convert corpus to a matrix of dimensions:
        (# of tokens in corpus, # sources)
        """
        return np.concatenate([
            utils._spans_to_array(
                doc,
                sources,
                self.label2idx,
                self.labels_without_prefix,
                self.prefixes if self.strict_match else None
            ) for doc in corpus
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


    def _covered_by_label(self, label_idx:int) -> np.ndarray:
        """Get indicator vector z where z_i = 1 if x_1 is labeled label_idx
        by at least one LF.
        """ 
        return (self._L_sparse == label_idx).max(axis=1)


    def _covered_by_label_counts(self, label_idx:int) -> np.ndarray:
        """Get count vector c where c_i is the # of times the ith source
        predicted a token to have the label value"""
        return np.ravel((self._L_sparse == label_idx).sum(axis=0))


    def _overlapped_data_points(self) -> np.ndarray:
        """Get indicator vector z where z_i = 1 if x_i i
        labeled by more than one LF."""
        return np.where(np.ravel((self._L_sparse != 0).sum(axis=1)) > 1, 1, 0)


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


    def _get_indices_for_lf_with_label(
        self,
        L_sparse: sparse.csr_matrix,
        lf_idx:int,
        label_idx:int
    ) -> Tuple[List[int], List[int]]:
        """ Get indices within sparse matrix that have label_idx for LF.
        """
        lf_vals = L_sparse.getcol(lf_idx)
        rows, cols, _ =  sparse.find(lf_vals == label_idx)
        return rows, cols