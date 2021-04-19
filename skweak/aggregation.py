
from abc import abstractmethod
from typing import Iterable, List, Set, Dict
import numpy as np
from .base import AbstractAnnotator
from spacy.tokens import Doc, Span  # type: ignore
from . import utils
import pickle
import hmmlearn
import hmmlearn.base
import pandas
import tempfile
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

####################################################################
# Aggregation models
####################################################################


class BaseAggregator(AbstractAnnotator):
    """Base aggregator to combine all labelling sources into a single annotation layer"""

    def __init__(self, name: str, labels: List[str], sequence_labelling: bool = True,
                 prefixes: str = "BIO"):
        """Creates a new token-level aggregator with the following arguments:
        - name is the aggregator name
        - labels is a list of output labels such as PERSON, ORG etc. 
        - If sequence_labelling is set to True, the labels are aggregated at the token-level
         (using IO/BIO/BILUO prefixes). Otherwise, skweak simply groups together spans with the 
          same (start,end) boundary, and aggregates their labels.
        - prefixes must be either 'IO', 'BIO', 'BILUO'. Ignored if sequence_labelling is False
        """

        super(BaseAggregator, self).__init__(name)

        if sequence_labelling:
            if prefixes not in {"IO", "BIO", "BILUO", "BILOU"}:
                raise RuntimeError(
                    "Tagging scheme must be 'IO', 'BIO', 'BILUO' or ''")
            self.out_labels = ["O"]
            for label in labels:
                for prefix in prefixes.replace("O", ""):
                    self.out_labels.append("%s-%s" % (prefix, label))
        else:
            self.out_labels = labels

        # We may also have "underspecified labels" that may stand for several
        # possible output labels (see below)
        self.underspecified_labels = {}

    def __call__(self, doc: Doc) -> Doc:
        """Aggregates all weak supervision sources"""

        if len(doc.spans) > 0:

            # Extracting the observation data
            df = self.get_observation_df(doc)

            # Running the actual aggregation
            agg_df = self._aggregate(df)

            if "O" in self.out_labels:
                # Converting back to spans or token labels
                output_spans = utils.token_array_to_spans(
                    agg_df.values, self.out_labels)
                output_probs = utils.token_array_to_probs(
                    agg_df.values, self.out_labels)
            else:
                output_spans = agg_df.idxmax(axis=1).to_dict()
                output_probs = {span: {label: prob for label, prob in distrib.items() if prob > 0.1}
                                for span, distrib in agg_df.to_dict(orient="index").items()}

            # Storing the results (both as spans and with the full probs)
            doc.spans[self.name] = [Span(doc, start, end, label=label)
                                    for (start, end), label in output_spans.items()]
            doc.spans[self.name].attrs["probs"] = output_probs
            doc.spans[self.name].attrs["aggregated"] = True
            doc.spans[self.name].attrs["sources"] = list(df.columns)

        return doc

    def get_observation_df(self, doc: Doc):
        """Returns a dataframe containing the observed predictions of each labelling
        sources for the document. The content of the dataframe depends on the prefixes.
        If prefixes was set to IO/BIO/BILUO, the dataframe has one row per token.
        If prefixes was set to False, the dataframe has one row per unique spans."""

        # Extracting the sources to consider (and filtering out the ones to avoid)
        sources = [source for source in doc.spans if len(doc.spans[source]) > 0
                   and "aggregated" not in doc.spans[source].attrs]

        # If the aggregation includes token-level segmentation, returns a dataframe
        # with token-level predictions
        if "O" in self.out_labels:
            data = utils.spans_to_array(doc, self.observed_labels, sources)
            return pandas.DataFrame(data, columns=sources)

        # Otherwise, returns a dataframe with span-level predictions
        else:
            # Extracts a list of unique spans (with identical boundaries)
            unique_spans = set((span.start, span.end)
                               for s in sources for span in doc.spans[s])
            sorted_spans = sorted(unique_spans)
            spans_indices = {span: i for i, span in enumerate(sorted_spans)}

            data = np.full((len(unique_spans), len(sources)),
                           fill_value=0, dtype=np.int16)

            # Populating the array with the labels from each source
            label_indices = {l: i for i, l in enumerate(self.observed_labels)}

            for source_index, source in enumerate(sources):
                for span in doc.spans[source]:
                    if span.label_ in self.observed_labels:
                        span_index = spans_indices[(span.start, span.end)]
                        data[span_index, source_index] = label_indices[span.label_]

            # We only consider spans with at least one concrete prediction
            masking = np.full(len(unique_spans), fill_value=True, dtype=bool)
            for i, row in enumerate(data):
                if row.max() <= 0 or row[row>0].min() > len(self.out_labels):
                    masking[i] = False
            data = data[masking]

            sorted_unique_spans = [span for i, span in enumerate(sorted_spans) if masking[i]]
            return pandas.DataFrame(data, columns=sources, index=sorted_unique_spans)

    @abstractmethod
    def _aggregate(self, observations: pandas.DataFrame) -> pandas.DataFrame:
        """Takes as input a 2D dataframe of shape (nb_entries, nb_sources) 
        associating each token/span to a set of observations from labelling 
        sources, and returns a 2D dataframe of shape (nb_entries, nb_labels)
        assocating each entry to the probability of each output label. 
        """

        raise NotImplementedError("must implement aggregate_spans")

    def add_underspecified_label(self, underspec_label: str, satisfied_values: Set[str]):
        """Specifies that the given label is not a specific output but a underspecified
        label that may be satisfied by several values. For instance, a source could label 
        a span as ENT (indicating that the span can receive any non-null label) 
        or NOT-PERSON (indicating that the label should not be a person)."""

        if "O" in self.out_labels:
            prefixes = {l.split("-", 1)[0]
                        for l in self.out_labels if "-" in l}
            for prefix in prefixes:
                underspec_label_with_prefix = "%s-%s" % (
                    prefix, underspec_label)
                prefixed_vals = {"%s-%s" % (prefix, value)
                                 for value in satisfied_values}
                self.underspecified_labels[underspec_label_with_prefix] = prefixed_vals
        else:
            self.underspecified_labels[underspec_label] = satisfied_values

    @property
    def observed_labels(self) -> List[str]:
        """Returns the possible labels that can be observed in labelling sources
        (that is, either "concrete" output labels or underspecified labels)."""

        observed = ["O"] if "O" not in self.out_labels else []
        observed += self.out_labels
        observed += sorted(self.underspecified_labels.keys())
        return observed

    def get_underspecification_matrix(self) -> np.ndarray:
        """Creates a boolean matrix of shape (nb_underspecified_labels, nb_out_labels)
        which specifies, for each underspecified label in prefix form (like B-ENT),
        the set of concrete output labels satisfying it (like B-ORG, B-PERSON,etc.)"""

        matrix = np.zeros((len(self.underspecified_labels),
                           len(self.out_labels)), dtype=bool)
        for i, underspec_label in enumerate(sorted(self.underspecified_labels)):
            for satisfied_label in self.underspecified_labels[underspec_label]:
                matrix[i, self.out_labels.index(satisfied_label)] = True
        return matrix


class MajorityVoter(BaseAggregator):
    """Simple aggregator based on majority voting"""

    def __init__(self, name: str, labels: List[str], sequence_labelling: bool = True,
                 initial_weights=None, prefixes: str = "BIO"):
        """Creates a majority voter to aggregate spans. Arguments:
        - name is the aggregator name
        - labels is a list of output labels such as PERSON, ORG etc. 
        - If sequence_labelling is set to True, the labels are aggregated at the token-level
         (using IO/BIO/BILUO prefixes). Otherwise, skweak simply groups together spans with the 
          same (start,end) boundary, and aggregates their labels.
        - initial_weights is a dictionary associating source names to numerical weights
          in the range [0, +inf]. The default assumes weights = 1 for all functions. You
          can disable a labelling function by giving it a weight of 0.
        - prefixes must be either 'IO', 'BIO', 'BILUO'. Ignored if sequence_labelling is False
        """

        super(MajorityVoter, self).__init__(
            name, labels, sequence_labelling, prefixes)
        self.weights = initial_weights if initial_weights else {}

    def _aggregate(self, obs: pandas.DataFrame, coefficient=0.1) -> pandas.DataFrame:
        """Takes as input a 2D dataframe of shape (nb_entries, nb_sources) 
        associating each token/span to a set of observations from labelling 
        sources, and returns a 2D dataframe of shape (nb_entries, nb_labels)
        assocating each entry to the probability of each output label. 

        This probability is here computed based on making each source "vote"
        on its output label. The most likely label will thus be the one that
        is indicated by most sources. If underspecified labels are included, they 
        are also part of the vote count. """

        # We count the votes for each label on all sources
        def count_function(x):
            # For sequence-labelling, we count the frequency of "O" labels (since "O"
            # is one possible output), but we can ignore it for classification
            min_val = 0 if "O" in self.out_labels else 1
            nb_obs_to_count = len(self.observed_labels)-min_val
            
            weights = [self.weights.get(source, 1) for source in obs.columns[x>=min_val]]
            return np.bincount(x[x>=min_val]-min_val, weights=weights, minlength=nb_obs_to_count) 

        label_votes = np.apply_along_axis(count_function, 1, obs.values)

        # For token-level segmentation (with a special O label), the number of "O" predictions
        # is typically much higher than any other predictions (since many labelling
        # sources are targeted for detecting specific labels). We thus need to
        # normalise the number of "O" predictions
        if "O" in self.out_labels:
            label_votes = label_votes.astype(np.float32)
            min_max = ((label_votes[:, 0] - label_votes[:, 0].min()) /
                       (label_votes[:, 0].max() - label_votes[:, 0].min() + 1E-20))
            label_votes[:, 0] = (min_max * len(obs.columns) * coefficient) + 1E-20
            

        # We start by counting only "concrete" (not-underspecified) labels
        out_label_votes = label_votes[:, :len(self.out_labels)]
        # We also add to the votes the values of underspecified labels
        if self.underspecified_labels:
            underspecified_label_votes = label_votes[:, len(self.out_labels):]
            additional_votes = underspecified_label_votes.dot(
                self.get_underspecification_matrix())
            out_label_votes += (additional_votes *
                                out_label_votes.astype(bool))

        # Normalisation
        total = np.expand_dims(out_label_votes.sum(axis=1), axis=1)
        probs = out_label_votes / total
        df = pandas.DataFrame(probs, index=obs.index, columns=self.out_labels)
        return df


class HMM(hmmlearn.base._BaseHMM, BaseAggregator):
    """Aggregator for labelled spans based on a HMM model. The HMM model is learned
    without access to the actual labels, using the Baum-Welch algorithm 
    (a special case of Expectation-Maximisation)"""

    def __init__(self, name: str, out_labels: List[str], sequence_labelling: bool = True,
                 prefixes: str = "BIO",  initial_weights=None, redundancy_factor=0.1):
        """Initialises the HMM model (which must be fitted before use). 
        Arguments:
        - name is the aggregator name
        - labels is a list of output labels such as PERSON, ORG etc. 
        - If sequence_labelling is set to True, the labels are aggregated at the token-level
         (using IO/BIO/BILUO prefixes). Otherwise, skweak simply groups together spans with the 
          same (start,end) boundary, and aggregates their labels.
        - prefixes must be either 'IO', 'BIO', 'BILUO'. Ignored if sequence_labelling is False
        - initial_weights is a dictionary associating source names to numerical weights
          in the range [0, +inf]. The default assumes weights = 1 for all functions. You
          can disable a labelling function by giving it a weight of 0.
        - redundancy_factor is the strength of the correlation-based weighting of each 
          labelling function. A value of 0.0 ignores redundancies"""

        BaseAggregator.__init__(self, name, out_labels,
                                sequence_labelling, prefixes)
        self.initial_weights = initial_weights if initial_weights else {}
        self.weights = dict(self.initial_weights)
        self.redundancy_factor = redundancy_factor

    def __call__(self, doc: Doc) -> Doc:
        """Aggregates all weak supervision sources (and fits the parameters if
        necessary)"""

        if not hasattr(self, "emit_probs"):
            return next(iter(self.fit_and_aggregate([doc])))
        else:
            return super(HMM, self).__call__(doc)

    def fit_and_aggregate(self, docs: Iterable[Doc], n_iter=4) -> Iterable[Doc]:
        """Starts by fitting the parameters of the HMM through Baum-Welch,
        then applies the resulting model to aggregate the outputs of the
        labelling functions."""

        self.fit(list(docs))
        return list(self.pipe(docs))

    def _aggregate(self, obs: pandas.DataFrame) -> pandas.DataFrame:
        """Takes as input a 2D dataframe of shape (nb_entries, nb_sources) 
        associating each token/span to a set of observations from labelling 
        sources, and returns a 2D dataframe of shape (nb_entries, nb_labels)
        assocating each entry to the probability of each output label. 

        This probability is here computed via a list of predicted labels 
        (extracted with Viterbi) for each token, along with the associated 
        probability according to the HMM model."""

        if not hasattr(self, "emit_probs"):
            raise RuntimeError("Model is not yet trained")

        # Convert the observations to one-hot representations
        X = {src: self._to_one_hot(obs[src].values) for src in obs.columns}

        # Compute the log likelihoods for each states
        framelogprob = self._compute_log_likelihood(X)

        # Run a forward pass
        _, forward_lattice = self._do_forward_pass(framelogprob)
        forward_lattice = forward_lattice - \
            forward_lattice.max(axis=1)[:, np.newaxis]

        # Transform into probabilities
        posteriors = np.exp(forward_lattice)
        posteriors = posteriors / \
            posteriors.sum(axis=1)[:, np.newaxis]  # type: ignore

        return pandas.DataFrame(posteriors, columns=self.out_labels, index=obs.index)

    def fit(self, docs: Iterable[Doc], cutoff: int = None, n_iter=4, tol=1e-2):
        """Train the HMM annotator based on a collection of documents 
        (which must have already been annotated using labelling functions)"""

        # We extract the docs
        if cutoff is None:
            docs = list(docs)
        else:
            docs = [doc for i, doc in enumerate(docs) if i <=cutoff]

        # We extract all source names
        sources = self._extract_sources(docs)
        if len(sources)== 0:
            raise RuntimeError("No document found with annotations")
        
        # Create uninformed priors to start with
        self._reset_counts(sources)

        # And add the counts from majority voter
        self._add_mv_counts(docs)

        # Finally, we postprocess the counts and get probabilities
        self._do_mstep()

        monitor = hmmlearn.base.ConvergenceMonitor(tol, n_iter, True)
        monitor._reset()
        for iter in range(n_iter):
            print("Starting iteration", (iter+1))
            curr_logprob = 0
            self._reset_counts(sources)

            # We loop on all documents at each iteration
            for doc_i, doc in enumerate(docs):

                # Transform the document annotations into observations
                obs = self.get_observation_df(doc)

                # Convert the observations to one-hot representations
                X = {src: self._to_one_hot(obs[src].values)
                     for src in obs.columns}

                # Compute its current log-likelihood
                framelogprob = self._compute_log_likelihood(X)
                # Make sure there is no token with no possible states

                if (np.isnan(framelogprob).any() or  # type: ignore
                        framelogprob.max(axis=1).min() < -100000):
                    pos = framelogprob.max(axis=1).argmin()
                    print("problem found for token", doc[pos], "in", self.name)
                    return

                # We run a forward and backward pass to compute the posteriors
                logprob, fwdlattice = self._do_forward_pass(framelogprob)
                curr_logprob += logprob
                bwdlattice = self._do_backward_pass(framelogprob)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice)

                # We accumulate the statistics in the counts
                self._accumulate_statistics(
                    X, framelogprob, posteriors, fwdlattice, bwdlattice)

                if doc_i > 0 and doc_i % 1000 == 0:
                    print("Number of processed documents:", doc_i)
            print("Finished E-step with %i documents" % len(docs))

            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            self._do_mstep()

            monitor.report(curr_logprob)
            if monitor.converged:
                break
        return self

    def _compute_log_likelihood(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """Computes the log likelihood for the observed sequence"""

        logsum = np.float32()
        for source in X:

            # Include the weights to the probabilities
            probs = (self.emit_probs[source]  # type:ignore
                     ** self.weights.get(source, 1))

            # We compute the likelihood of each state given the observations
            probs = np.dot(X[source], probs.T)

            # Impossible states have a logprob of -inf
            log_probs = np.ma.log(probs).filled(-np.inf)

            logsum += log_probs

        # We also add a constraint that the probability of a state is zero
        # if no labelling functions observes it
        X_all_obs = np.zeros(logsum.shape, dtype=bool)  # type: ignore
        for source in self.emit_counts:
            if source in X:
                if "O" in self.out_labels:
                    X_all_obs += X[source][:, :len(self.out_labels)]
                else:
                    X_all_obs += X[source][:, 1:len(self.out_labels)+1]
        logsum = np.where(X_all_obs, logsum, -np.inf)

        return logsum  # type: ignore

    def _extract_sources(self, docs: Iterable[Doc], max_number=1000):
        """Extract the names of all labelling sources mentioned in the documents
        (and not included in the list of sources to avoid)"""
        sources = set()
        for i, doc in enumerate(docs):
            for source in doc.spans:
                if (len(doc.spans[source]) > 0 and
                        "aggregated" not in doc.spans[source].attrs):
                    sources.add(source)
            if i > max_number:
                break

        return sources

    def _reset_counts(self, sources, fp_prior=0.1, fn_prior=0.3, concentration=1):
        """Reset the various counts/statistics used for for the M-steps, and also
        adds uninformed priors for the start, transition and emission counts"""

        nb_labels = len(self.out_labels)
        nb_obs = len(self.observed_labels)

        # We reset all counts
        self.start_counts = np.zeros(shape=(nb_labels,))
        self.trans_counts = np.zeros(shape=(nb_labels, nb_labels))
        self.emit_counts = {source: np.zeros(
            shape=(nb_labels, nb_obs)) for source in sources}

        self.corr_counts = {}
        for src1 in sources:
            for src2 in self._get_correlated_sources(src1, sources):
                self.corr_counts[(src1, src2)] = np.zeros(shape=(nb_obs, nb_obs))

        # We add some prior values
        self.start_counts += concentration + 1E-10
        self.trans_counts += concentration + 1E-10
        for source in sources:
            if "O" in self.out_labels:
                self.emit_counts[source][:, :nb_labels] = np.eye(nb_labels)  * concentration
                self.emit_counts[source][0, :] += fp_prior * concentration
            else:
                self.emit_counts[source][:, 1:nb_labels+1] = np.eye(nb_labels)  * concentration
            self.emit_counts[source][:, 0] += fn_prior * concentration
            self.emit_counts[source] += 0.000001
                

        for source, source2 in self.corr_counts:
            self.corr_counts[(source, source2)] = np.eye(nb_obs)  * concentration
            self.corr_counts[(source, source2)][:, 0] += fn_prior * concentration
            self.corr_counts[(source, source2)][0, :] += fp_prior * concentration
            self.corr_counts[(source, source2)] += 0.000001

    def _get_correlated_sources(self, source, all_sources):
        """Extracts the list of possible correlated sources, according to specific
        naming conventions. This method can be modified/replaced if need."""

        source2 = source.replace("_cased", "").replace("_uncased", "")
        corr_sources = []
        for other_source in all_sources:
            if other_source == source:
                continue
            other_source2 = other_source.replace(
                "_cased", "").replace("_uncased", "")
            if source2 == other_source2:
                corr_sources.append(other_source)
            elif (source2.startswith(other_source2) or source2.endswith(other_source2)
                  or other_source2.startswith(source2) or other_source2.endswith(source2)):
                corr_sources.append(other_source)
        return corr_sources

    def _add_mv_counts(self, docs: Iterable[Doc]):
        """Getting initial counts for the HMM parameters based on an ensemble of
        majority voters"""

        # We rely on an ensemble majority voter to get the first counts
        # The weights of each source in the majority voter are based on
        # the initial weights of the HMM, adjusted with an upper bound on
        # the redundancy of each source
        init_mv_weights = {source: self.initial_weights.get(source, 1) *
                           np.exp(-self.redundancy_factor *  # type: ignore
                                  len([s2 for s1, s2 in self.corr_counts if s1 == source]))
                           for source in self.emit_counts}
        mv = MajorityVoter("", self.out_labels, sequence_labelling=False,
                           initial_weights=init_mv_weights)
        mv.underspecified_labels = self.underspecified_labels

        for doc in docs:

            # We extract the observations
            obs = self.get_observation_df(doc)

            # And aggregate the results
            agg_array = mv._aggregate(obs).values

            # Update the start probabilities
            self.start_counts += agg_array[0, :]

            # Update the transition probabilities
            for i in range(1, len(agg_array)):
                self.trans_counts += np.outer(agg_array[i-1], agg_array[i])

            # Get indicator matrices for the observations
            one_hots = {src: self._to_one_hot(
                obs[src].values) for src in obs.columns}

            # Update the emission probabilities
            for source in one_hots:
                mv_counts = np.dot(agg_array.T, one_hots[source])
                self.emit_counts[source] += mv_counts

            for src, src2 in self.corr_counts:
                if src in one_hots and src2 in one_hots:
                    self.corr_counts[(
                        src, src2)] += np.dot(one_hots[src2].T, one_hots[src])

    def _to_one_hot(self, vector: np.ndarray) -> np.ndarray:
        """Given a vector of indices to observed labels, returns a 2D
        boolean matrix representing the presence/absence of a label. """

        matrix = np.zeros((vector.size, len(self.observed_labels)), dtype=bool)
        matrix[np.arange(vector.size), vector] = True
        return matrix

    def _accumulate_statistics(self, X: Dict[str, np.ndarray], framelogprob: np.ndarray,
                               posteriors: np.ndarray, fwdlattice, bwdlattice):
        """Acccumulate the counts based on the sufficient statistics"""

        # Update the start counts
        self.start_counts += posteriors[0]

        # Updating the transition counts
        n_samples, n_components = framelogprob.shape
        if n_samples > 1:
            log_xi_sum = np.full((n_components, n_components), -np.inf)
            hmmlearn._hmmc._compute_log_xi_sum(n_samples, n_components, fwdlattice,  # type: ignore
                                               hmmlearn.base.log_mask_zero(
                                                   self.transmat_),
                                               bwdlattice, framelogprob, log_xi_sum)
            self.trans_counts += np.exp(log_xi_sum)

        # Updating the emission counts
        for src in X:
            self.emit_counts[src] += np.dot(posteriors.T, X[src])

        for src, src2 in self.corr_counts:
            if src in X and src2 in X:
                self.corr_counts[(src, src2)] += np.dot(X[src2].T, X[src])

    def _do_mstep(self):
        """Performs the maximisation step of the EM algorithm"""

        # We do some postprocessing of the counts to erase invalid counts
        if "O" in self.out_labels:
            self._postprocess_counts()

        # We normalise to get probabilities
        self.startprob_ = self.start_counts / \
            (self.start_counts.sum() + 1E-100)

        trans_norm = (self.trans_counts.sum(axis=1) + 1E-100)[:, np.newaxis]
        self.transmat_ = self.trans_counts / trans_norm

        self.emit_probs: Dict[str, np.ndarray] = {}
        for source in self.emit_counts:
            normalisation = (
                self.emit_counts[source] + 1E-100).sum(axis=-1)[:, np.newaxis]
            self.emit_probs[source] = self.emit_counts[source] / normalisation

        # Compute weights per labelling sources
        self._update_weights()

    def _update_weights(self):
        """Update the weights of each labelling function to account for correlated sources"""

        # We reset the weights
        self.weights = {}
        for source in self.emit_counts:
            init_weight = self.initial_weights.get(source, 1)
            self.weights[source] = np.full(fill_value=init_weight,
                                           shape=len(self.observed_labels),
                                           dtype=np.float32)

        # We also take into account redundancies with other labelling functions
        for source in self.emit_counts:
            for i in range(len(self.observed_labels)):

                # We compute the recall with each correlated source
                recalls_with_corr_sources = []
                for (source1, source2), counts in self.corr_counts.items():
                    if source1 == source:
                        recall = counts[i, i]/counts[i, :].sum()
                        recalls_with_corr_sources.append(recall)

                # The weight decreases with the number of correlated labelling functions
                # that have a high recall with the current function
                self.weights[source][i] *= np.exp(-self.redundancy_factor *  # type: ignore
                                                  np.sum(recalls_with_corr_sources))

    def _postprocess_counts(self):
        """Postprocess the counts to erase invalid starts, transitions or emissions"""

        prefixes = {label.split("-", 1)[0] for label in self.out_labels}

        # We make sure the counts for invalid starts (i.e. "L-ORG") are zero
        for i, label in enumerate(self.out_labels):
            if not utils.is_valid_start(label, prefixes):
                self.start_counts[i] = 0

        # We make sure the counts for invalid transitions (i.e. B-ORG -> I-GPE) are zero
        for i, label in enumerate(self.out_labels):
            for j, label2 in enumerate(self.out_labels):
                if not utils.is_valid_transition(label, label2, prefixes):
                    self.trans_counts[i, j] = 0

        # We also take into account the underspecified label matrix (but in a soft manner)
        for emit_counts in self.emit_counts.values():
            cur_counts = emit_counts[:, len(self.out_labels):]
            new_counts = 0.1 * cur_counts + 0.9 * cur_counts * \
                self.get_underspecification_matrix().T
            emit_counts[:, len(self.out_labels):] = new_counts

    def pretty_print(self, sources=None, nb_digits=2, show_counts=False):
        """Prints out a summary of the HMM models"""

        import pandas
        pandas.set_option("display.width", 1000)
        valid_sources = [source for source in self.emit_counts
                         if self.initial_weights.get(source, 1) > 0]
        print("HMM model with labelling functions:", valid_sources)
        print("Output labels:", self.out_labels)
        if self.underspecified_labels:
            print("Underspecified labels:", self.underspecified_labels)
        print("--------")
        if hasattr(self, "startprob_"):
            print("Start distribution:")
            print(pandas.Series(self.startprob_, index=self.out_labels).round(nb_digits))
            print("--------")
        if hasattr(self, "transmat_"):
            print("Transition model:")
            print(pandas.DataFrame(self.transmat_, index=self.out_labels,
                               columns=self.out_labels).round(nb_digits))
            print("--------")
        for source in sorted(self.emit_counts):
            if self.initial_weights.get(source, 1) == 0.0:
                continue
            if sources == None or source in sources:
                print("Emission model for: %s" % (source))
                vals = (self.emit_counts if show_counts else self.emit_probs)[source]
                df = pandas.DataFrame(vals, index=self.out_labels,
                                      columns=self.observed_labels)
                if self.weights:
                    dft = df.transpose()
                    dft["weights"] = self.weights[source]
                    df = dft.transpose()
                print(df.round(nb_digits))
                print("--------")

    def save(self, filename):
        """Saves the HMM model to a file"""
        fd = open(filename, "wb")
        pickle.dump(self, fd)
        fd.close()

    @classmethod
    def load(cls, pickle_file):
        """Loads the model from an existing file"""
        print("Loading", pickle_file)
        fd = open(pickle_file, "rb")
        ua = pickle.load(fd)
        fd.close()
        return ua
