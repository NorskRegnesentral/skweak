

import numpy as np
from abc import abstractmethod
from typing import Iterable, List, Optional, Set, Dict, Tuple, Union
from .aggregation import AbstractAggregator, SequenceAggregatorMixin
from .aggregation import TextAggregatorMixin, MultilabelAggregatorMixin
from .voting import MajorityVoterMixin
from spacy.tokens import Doc, Span  # type: ignore
from . import utils, base
import hmmlearn
import hmmlearn.base
import pandas
import scipy.special

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##########################################
# Core class for all generative models
##########################################


class GenerativeModelMixin(AbstractAggregator):
    """Implementation of a subset of methods from AbstractAggregator when
    the aggregation method is based on a generative model where the states
    correspond to the "true" (latent) labels, and the observations to the
    predictions of the labelling sources.
    
    This class should not be instantiated directly.
    """
    
    def __init__(self, initial_weights:Optional[Dict[str,float]]=None, 
                 redundancy_factor:float=0.1):
        """Creates a new aggregator based on a generative model. Arguments:
        - initial_weights is a dictionary associating source names to numerical weights
          in the range [0, +inf]. The default assumes weights = 1 for all functions. You
          can disable a labelling function by giving it a weight of 0.
        - redundancy_factor is the strength of the correlation-based weighting of each 
        labelling function. A value of 0.0 ignores redundancies"""
       
        self.initial_weights = initial_weights if initial_weights else {}
        self.weights = dict(self.initial_weights)
        self.redundancy_factor = redundancy_factor
        
        self.emit_counts = {}
        self.corr_counts = {}
    
    
    def aggregate(self, obs: pandas.DataFrame) -> pandas.DataFrame:
        """Takes as input a 2D dataframe of shape (nb_entries, nb_sources) 
        associating each token/span to a set of observations from labelling 
        sources, and returns a 2D dataframe of shape (nb_entries, nb_labels)
        assocating each entry to the probability of each output label. """

        if not hasattr(self, "emit_probs"):
            raise RuntimeError("Model is not yet trained")
        
        # Convert the observations to one-hot representations
        X = self.to_one_hots(obs)
        
        # Compute the posteriors
        posteriors = self.get_posteriors(X)

        return pandas.DataFrame(posteriors, columns=self.out_labels, index=obs.index)

    @abstractmethod
    def get_posteriors(self, X: Dict[str,np.ndarray]) -> np.ndarray:
        
        """Given a dictionary mapping labelling sources to boolean 2D matrixes
        (expressing the labels predicted by the source for each data point, in
        one-hot encoding format), returns the posterior distributions on the 
        output labels. The method first computes the log-likelihood of the observations,
        then runs forward and backward passes to infer the posteriors."""
       
        raise NotImplementedError("Must implement get_posteriors")
              

    def _fit(self, all_obs: Iterable[pandas.DataFrame], 
            cutoff: int = None, n_iter=4, tol=1e-2):
        """Train the HMM annotator based on a collection of observations from 
        labelling sources)"""

        # We extract the observations
        all_obs = [obs for i, obs in enumerate(all_obs) if len(obs.columns) > 0 
                   and (cutoff is None or i <=cutoff)]
    
        # We extract the possible labelling sources
        sources = {source for obs in all_obs for source in obs.columns}
        if len(sources)== 0:
            raise RuntimeError("No document found with annotations")
        
        # Create uninformed priors to start with
        self._reset_counts(sources)

        # And add the counts from majority voter
        self._add_mv_counts(all_obs)

        # Finally, we postprocess the counts and get probabilities
        self._do_mstep_latent()
        self._do_mstep_emissions()

        monitor = hmmlearn.base.ConvergenceMonitor(tol, n_iter, True)
        monitor._reset()
        for iter in range(n_iter):
            print("Starting iteration", (iter+1))
            curr_logprob = 0
            self._reset_counts(sources)

            # We loop on all observations at each iteration
            for obs_i, obs in enumerate(all_obs):
                
                # Special case: no observations from any source
                if len(obs.columns)==0:
                    continue

                # Convert the observations to one-hot representations
                X = self.to_one_hots(obs)

                # Update the sufficient statistics based on the observations
                curr_logprob += self._accumulate_statistics(X)

                if obs_i > 0 and obs_i % 1000 == 0:
                    print("Number of processed documents:", obs_i)
            print("Finished E-step with %i documents" % len(all_obs))

            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            self._do_mstep_latent()
            self._do_mstep_emissions()

            monitor.report(curr_logprob)
            if monitor.converged:
                break
        return self

    def _get_log_likelihood(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """Computes the log likelihood for the observed sequence"""

        logsum = np.float32(0.0) #type:ignore
        
        for source in X:

            # Include the weights to the probabilities
            probs = (self.emit_probs[source]  # type:ignore
                     ** self.weights.get(source, 1))

            # We compute the likelihood of each state given the observations
            likelihoods = np.dot(X[source], probs.T)

            # We ignore cases where the source does not predict anything
            likelihoods[X[source].sum(axis=1)==0] = 1.0 #type: ignore
            
            # Impossible states have a logprob of -inf
            log_likelihoods = np.ma.log(likelihoods).filled(-np.inf)
            logsum += log_likelihoods #type:ignore

        # We also add a constraint that the probability of a state is 
        # zero if no labelling functions "votes" for it
        votes = np.sum(X[source].dot(self._get_vote_matrix(include_underspec=False)) 
                       for source in self.emit_counts if source in X)
        logsum= np.where(votes > 0, logsum, -np.inf)

        if (np.isnan(logsum).any() or # type: ignore  
                logsum.max(axis=1).min() < -100000): # type: ignore  
            pos = logsum.max(axis=1).argmin() # type: ignore  
            raise RuntimeError("No valid state found at position %i"%pos)
                
        return logsum  # type: ignore


    @abstractmethod
    def _reset_counts(self, sources):
        """Resets the counts for model parameters"""
        
        raise NotImplementedError("must implement _reset_counts")


    def _get_correlated_sources(self, source, all_sources):
        """Extracts the list of possible correlated sources, according to specific
        naming conventions. This method can be modified/replaced if needed."""

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

    @ abstractmethod
    def _add_mv_counts(self, all_obs: Iterable[pandas.DataFrame]):
        """Getting initial counts for the generative model parameters 
        based on a majority voter"""
        raise NotImplementedError("must implement _add_mv_counts")


    def to_one_hots(self, obs:pandas.DataFrame) -> Dict[str, np.ndarray]:
        """Given a dataframe of observations (each row corresponding to a 
        token/span, each column to a labelling source, and each cell to the
        index of label predicted by the source for that token/span), returns
        a dictionary mapping the name of each labelling source to a dictionary
        mapping each labelling source to a 2D boolean matrix representing 
        the presence/absence of a label."""
        
        one_hots_dic = {}
        for source in obs.columns:
            vector = obs[source].values #type: ignore
            matrix = np.zeros((vector.size, #type: ignore
                               len(self.observed_labels)), dtype=bool)
            for i, j in enumerate(vector): #type: ignore
                if j >= 0:
                    matrix[i,j] = True
            one_hots_dic[source] = matrix
        return one_hots_dic
        
    @abstractmethod
    def _accumulate_statistics(self, X: Dict[str, np.ndarray]) -> float:
        """Acccumulate the counts for the sufficient statistics of the generative
        model based on the provided observations. Returns the logprob"""
        raise NotImplementedError("must implement _accumulate_statistics")


    @abstractmethod
    def _do_mstep_latent(self):
        """Performs the maximisation step of the EM algorithm for the latent part
        of the model (i.e. the prior probabilities for a Naive Bayes model, or the
        start and transition probabilities for an HMM model)"""
        
        raise NotImplementedError("must implement _do_mstep_latent")

    def _do_mstep_emissions(self):
        """Performs the maximisation step of the EM algorithm for the emission
        models"""

        self.emit_probs: Dict[str, np.ndarray] = {}
        for source in self.emit_counts:
            normalisation = (
                self.emit_counts[source] + 1E-100).sum(axis=-1)[:, np.newaxis]
            self.emit_probs[source] = self.emit_counts[source] / normalisation

        # Compute weights per labelling sources
        self._update_weights()        

    def _update_weights(self):
        """Update the weights of each labelling function to account for 
        correlated sources"""

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


    def _pretty_print_emissions(self, sources:Optional[List[str]]=None, 
                                nb_digits:int=2, show_counts:bool=False):
        """Prints the emission models for the various labelling sources.
        Arguments:
        - sources is the list of the labelling sources to display. If set to
          None, prints the emission models for all sources
        - nb_digits is the number of digits to include in the tables
        - show_counts: if true, displays the raw counts instated of the normalised
          probabilities"""

        valid_sources = [source for source in self.emit_counts
                         if self.initial_weights.get(source, 1) > 0]
        print("Labelling functions in model:", valid_sources)
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



###############################################
# Naive Bayes class for text classification
###############################################

class NaiveBayes(GenerativeModelMixin, TextAggregatorMixin):
    """Aggregator based on a Naive Bayes model for text classification. The 
    parameters of the Naive Bayes model are learned without access to the 
    actual labels using Expectation-Maximisation."""
    
    def __init__(self, name:str, labels:List[str], 
                 prior_probs:Optional[Dict[str,float]]=None,
                 initial_weights:Optional[Dict[str,float]]=None,
                 redundancy_factor=0.1):
        """Creates a new aggregator with a HMM model. Arguments:
        - name is the aggregator name
        - labels is a list of output labels to aggregate. Labels that are not 
          mentioned here are ignored. 
        - prior_probs is a dictionary associating output labels to prior state 
          probabilities. The default assumes uniform prior probabilities.
       - initial_weights is a dictionary associating source names to numerical weights
          in the range [0, +inf]. The default assumes weights = 1 for all functions. You
          can disable a labelling function by giving it a weight of 0.
        - redundancy_factor is the strength of the correlation-based weighting of each 
          labelling function. A value of 0.0 ignores redundancies"""
       
        AbstractAggregator.__init__(self, name, labels)
        GenerativeModelMixin.__init__(self, initial_weights, redundancy_factor)
        
        if prior_probs:
            if any(l for l in self.out_labels if l not in prior_probs 
                   or prior_probs[l]<0 or prior_probs[l]>1):
                raise RuntimeError("Prior probabilities not well-formed")
            elif abs(sum(prior_probs.values()) - 1.0) > 0.01:
                 raise RuntimeError("Prior probabilities not well-formed")
           
            self.prior_probs = np.array([prior_probs[l] for l in self.out_labels])
        else:
            self.prior_probs = np.array([1/len(self.out_labels)]*len(self.out_labels))

    def get_posteriors(self, X: Dict[str,np.ndarray]) -> np.ndarray:
        
        """Given a dictionary mapping labelling sources to boolean 2D matrixes
        (expressing the labels predicted by the source for each data point, in
        one-hot encoding format), returns the posterior distributions on the 
        output labels. The method first computes the log-likelihood of the observations,
        then runs forward and backward passes to infer the posteriors."""
       
        # Compute its current log-likelihood
        framelogprob = self._get_log_likelihood(X)
        
        # We add the log of the prior state probabilities
        posteriors = np.exp(framelogprob + np.log(self.prior_probs).T) #type: ignore
        
        # And normalise
        norm = (posteriors.sum(axis=1) + 1E-100)[:, np.newaxis] #type: ignore
        posteriors = posteriors / norm
        
        return posteriors #type: ignore
              

    def _reset_counts(self, sources, fp_prior=0.1, fn_prior=0.3, concentration=10):
        """Reset the emission (and correlation) counts/statistics to uninformed priors"""

        nb_labels = len(self.out_labels)
        nb_obs = len(self.observed_labels)

        # We reset all counts
        self.state_counts = np.zeros(shape=(nb_labels,))
        self.state_counts += concentration + 1E-10        
        self.emit_counts = {source: np.zeros(
            shape=(nb_labels, nb_obs)) for source in sources}

        self.corr_counts = {}
        for src1 in sources:
            for src2 in self._get_correlated_sources(src1, sources):
                self.corr_counts[(src1, src2)] = np.zeros(shape=(nb_obs, nb_obs))

        # We add some prior values
        for source in sources:
            for i, label in enumerate(self.out_labels):
                if label in self.observed_labels:
                    self.emit_counts[source][i, self.observed_labels.index(label)] = concentration
            self.emit_counts[source] += 0.000001      

        for source, source2 in self.corr_counts:
            self.corr_counts[(source, source2)] = np.eye(nb_obs)  * concentration
            self.corr_counts[(source, source2)][:, 0] += fn_prior * concentration
            self.corr_counts[(source, source2)][0, :] += fp_prior * concentration
            self.corr_counts[(source, source2)] += 0.000001
                          

    def _add_mv_counts(self, all_obs: Iterable[pandas.DataFrame]):
        """Getting initial counts for the HMM parameters based on a majority voter"""

        # We rely on a majority voter to get the first counts
        # The weights of each source in the majority voter are based on
        # the initial weights of the HMM, adjusted with an upper bound on
        # the redundancy of each source
        init_mv_weights = {source: self.initial_weights.get(source, 1) *
                           np.exp(-self.redundancy_factor *  # type: ignore
                                  len([s2 for s1, s2 in self.corr_counts if s1 == source]))
                           for source in self.emit_counts}
        mv = MajorityVoterMixin(initial_weights=init_mv_weights)
        mv.out_labels = self.out_labels
        mv.observed_labels = self.observed_labels
        mv.label_groups = self.label_groups
        for obs in all_obs:

            # And aggregate the results
            agg_array = mv.aggregate(obs).values

            if len(agg_array)==0:
                continue
            
            # Update the state counts
            self.state_counts += agg_array.sum(axis=0)

            # Get indicator matrices for the observations
            one_hots = self.to_one_hots(obs)

            # Update the emission counts
            for source in one_hots:
                mv_counts = np.dot(agg_array.T, one_hots[source])
                self.emit_counts[source] += mv_counts

            for src, src2 in self.corr_counts:
                if src in one_hots and src2 in one_hots:
                    self.corr_counts[(
                        src, src2)] += np.dot(one_hots[src2].T, one_hots[src])
        

    def _accumulate_statistics(self, X: Dict[str, np.ndarray]) -> float:
        """Acccumulate the counts based on the sufficient statistics"""
        
        # Compute its current log-likelihood
        framelogprob = self._get_log_likelihood(X)
        
        # We add the the prior state log-probabilities
        posteriors = np.exp(framelogprob + np.log(self.prior_probs.T)) #type:ignore
        
        # Update the start counts
        self.state_counts += posteriors.sum(axis=0) #type:ignore

        # Updating the emission counts
        for src in X:
            self.emit_counts[src] += np.dot(posteriors.T, X[src]) #type:ignore

        for src, src2 in self.corr_counts:
            if src in X and src2 in X:
                self.corr_counts[(src, src2)] += np.dot(X[src2].T, X[src])
                
        return scipy.special.logsumexp(posteriors,axis=1).sum() #type: ignore
        

    def _do_mstep_latent(self):
        """Performs the maximisation step of the EM algorithm for the latent
        part of the Naive Bayes model (prior state probabilities)"""
        
        self.prior_probs = (self.state_counts / self.state_counts.sum() + 1E-100)
     

    def pretty_print(self, sources=None, nb_digits=2):
        """Prints out a summary of the Naive Bayes model"""

        import pandas
        pandas.set_option("display.width", 1000)
        print("Naive Bayes model with following parameters:")
        print("Output labels:", self.out_labels)
        if self.label_groups:
            print("Label groups:", self.label_groups)
        print("--------")
        
        if hasattr(self, "prior_probs"):
            print("Prior state distribution:")
            print(pandas.Series(self.prior_probs, index=self.out_labels).round(nb_digits))
            print("--------")
            
        self._pretty_print_emissions(sources, nb_digits)


###############################################
# HMM model for sequence labelling
###############################################


class HMM(GenerativeModelMixin,SequenceAggregatorMixin):
    """Aggregator for sequence labelling based on a HMM model. The parameters of
    the HMM model are learned without access to the actual labels, using the 
    Baum-Welch algorithm (a special case of Expectation-Maximisation)"""
    
    def __init__(self, name:str, labels:List[str], prefixes:str="BIO", 
                 initial_weights:Optional[Dict[str,float]]=None,
                 redundancy_factor=0.1):
        """Creates a new aggregator with a HMM model. Arguments:
        - name is the aggregator name
        - labels is a list of output labels to aggregate. Labels that are not 
          mentioned here are ignored. 
        - prefixes is the token-level tagging scheme, such as IO, BIO or BILUO      
        - initial_weights is a dictionary associating source names to numerical weights
          in the range [0, +inf]. The default assumes weights = 1 for all functions. You
          can disable a labelling function by giving it a weight of 0.
        - redundancy_factor is the strength of the correlation-based weighting of each 
          labelling function. A value of 0.0 ignores redundancies"""
                        
        AbstractAggregator.__init__(self, name, labels)
        SequenceAggregatorMixin.__init__(self, prefixes)
        GenerativeModelMixin.__init__(self, initial_weights, redundancy_factor)
        
        # Creates a new HMM object from hmmlearn (mainly used for their 
        # efficient implementation of forward and backward passes) 
        self.hmm = hmmlearn.base._BaseHMM()
    
            
    def get_posteriors(self, X: Dict[str,np.ndarray]) -> np.ndarray:
        
        """Given a dictionary mapping labelling sources to boolean 2D matrixes
        (expressing the labels predicted by the source for each data point, in
        one-hot encoding format), returns the posterior distributions on the 
        output labels. The method first computes the log-likelihood of the observations,
        then runs forward and backward passes to infer the posteriors."""
       
        # Compute its current log-likelihood
        framelogprob = self._get_log_likelihood(X)
                
        # We run a forward and backward pass to compute the posteriors
        _, fwdlattice = self.hmm._do_forward_log_pass(framelogprob)
        bwdlattice = self.hmm._do_backward_log_pass(framelogprob)
        posteriors = self.hmm._compute_posteriors_log(fwdlattice, bwdlattice)  
        
        return posteriors #type: ignore
              
    def _reset_counts(self, sources, fp_prior=0.1, fn_prior=0.3, concentration=1):
        """Reset the emission (and correlation) counts/statistics to uninformed priors"""

        nb_labels = len(self.out_labels)
        nb_obs = len(self.observed_labels)

        # We reset all counts
        self.start_counts = np.zeros(shape=(nb_labels,))
        self.trans_counts = np.zeros(shape=(nb_labels, nb_labels))
        self.emit_counts = {source: np.zeros(shape=(nb_labels, nb_obs))
                            for source in sources} 
        
        # We also count correlations between sources
        self.corr_counts = {}
        for src1 in sources:
            for src2 in self._get_correlated_sources(src1, sources):
                self.corr_counts[(src1, src2)] = np.zeros(shape=(nb_obs, nb_obs))
                           
        # We add some prior values to the start and transition counts
        self.start_counts += concentration + 1E-10
        self.trans_counts += concentration + 1E-10

        # And to the emission and correlation counts
        for source in sources:
            for i, label in enumerate(self.out_labels):
                if label in self.observed_labels:
                    self.emit_counts[source][i, self.observed_labels.index(label)] = concentration
            self.emit_counts[source][0, :] += fp_prior * concentration
            self.emit_counts[source][:, 0] += fn_prior * concentration
            self.emit_counts[source] += 0.000001     

        for source, source2 in self.corr_counts:
            self.corr_counts[(source, source2)] = np.eye(nb_obs)  * concentration
            self.corr_counts[(source, source2)][:, 0] += fn_prior * concentration
            self.corr_counts[(source, source2)][0, :] += fp_prior * concentration
            self.corr_counts[(source, source2)] += 0.000001



    def _add_mv_counts(self, all_obs: Iterable[pandas.DataFrame]):
        """Getting initial counts for the HMM parameters based on an ensemble of
        majority voters"""

        # We rely on majority voter to get the first counts
        # The weights of each source in the majority voter are based on
        # the initial weights of the HMM, adjusted with an upper bound on
        # the redundancy of each source
        init_mv_weights = {source: self.initial_weights.get(source, 1) *
                           np.exp(-self.redundancy_factor *  # type: ignore
                                  len([s2 for s1, s2 in self.corr_counts if s1 == source]))
                           for source in self.emit_counts}
        mv = MajorityVoterMixin(initial_weights=init_mv_weights)
        mv.out_labels = self.out_labels
        mv.observed_labels = self.observed_labels
        mv.label_groups = self.label_groups

        for obs in all_obs:

            # And aggregate the results
            agg_array = mv.aggregate(obs).values

            if len(agg_array)==0:
                continue
            
            # Update the start probabilities
            self.start_counts += agg_array[0, :]

            # Update the transition probabilities
            for i in range(1, len(agg_array)):
                self.trans_counts += np.outer(agg_array[i-1], agg_array[i])

            # Get indicator matrices for the observations
            one_hots = self.to_one_hots(obs)

            # Update the emission probabilities
            for source in one_hots:
                mv_counts = np.dot(agg_array.T, one_hots[source])
                self.emit_counts[source] += mv_counts

            for src, src2 in self.corr_counts:
                if src in one_hots and src2 in one_hots:
                    self.corr_counts[(
                        src, src2)] += np.dot(one_hots[src2].T, one_hots[src])
        

    def _accumulate_statistics(self, X: Dict[str, np.ndarray]):
        """Acccumulate the counts based on the sufficient statistics, and returns"""
        
        # Compute its current log-likelihood
        framelogprob = self._get_log_likelihood(X)
                
        # We run a forward and backward pass to compute the posteriors
        logprob, fwdlattice = self.hmm._do_forward_log_pass(framelogprob)
        bwdlattice = self.hmm._do_backward_log_pass(framelogprob)
        posteriors = self.hmm._compute_posteriors_log(fwdlattice, bwdlattice)  
        
        # Update the start counts
        self.start_counts += posteriors[0] #type: ignore

        # Updating the transition counts
        n_samples, n_components = framelogprob.shape
        if n_samples > 1:
            log_xi_sum = hmmlearn._hmmc.compute_log_xi_sum(fwdlattice, hmmlearn.base.log_mask_zero(self.hmm.transmat_),
                                                        bwdlattice, framelogprob)
            self.trans_counts += np.exp(log_xi_sum) #type: ignore

        # Updating the emission counts
        for src in X:
            self.emit_counts[src] += np.dot(posteriors.T, X[src]) #type: ignore

        for src, src2 in self.corr_counts:
            if src in X and src2 in X:
                self.corr_counts[(src, src2)] += np.dot(X[src2].T, X[src])
        
        return logprob

    def _do_mstep_latent(self):
        """Performs the maximisation step of the EM algorithm for the latent
        part of the HMM model (start and transition probabilities)"""
        
        self._do_sequence_labelling_checks()

        # We normalise to get probabilities
        self.hmm.startprob_ = self.start_counts / (self.start_counts.sum() + 1E-100)

        trans_norm = (self.trans_counts.sum(axis=1) + 1E-100)[:, np.newaxis]
        self.hmm.transmat_ = self.trans_counts / trans_norm


    def _do_sequence_labelling_checks(self):
        """Perform additional checks on the start, transition and
        emission counts to remove any invalid counts"""
        
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
   


    def pretty_print(self, sources=None, nb_digits=2):
        """Prints out a summary of the HMM models"""

        import pandas
        pandas.set_option("display.width", 1000)
        print("HMM model with following parameters:")
        print("Output labels:", self.out_labels)
        if self.label_groups:
            print("Label groups:", self.label_groups)
        print("--------")
        if hasattr(self.hmm, "startprob_"):
            print("Start distribution:")
            print(pandas.Series(self.hmm.startprob_, index=self.out_labels).round(nb_digits))
            print("--------")
        if hasattr(self.hmm, "transmat_"):
            print("Transition model:")
            print(pandas.DataFrame(self.hmm.transmat_, index=self.out_labels, #type: ignore
                               columns=self.out_labels).round(nb_digits))
            print("--------")
            
        self._pretty_print_emissions(sources, nb_digits)

       
class MultilabelNaiveBayes(MultilabelAggregatorMixin, NaiveBayes,
                            TextAggregatorMixin,AbstractAggregator):
    
    def __init__(self, name:str, labels:List[str], 
                 initial_weights:Optional[Dict[str,float]]=None):
        """Creates a new, multilabel aggregator for text classification using 
        Naive Bayes. The aggregation method is based on a generative model 
        where the states correspond to the "true" (latent) labels, and the 
        observations to the predictions of the labelling sources.
        
        Arguments:
        - name is the aggregator name
        - labels is a list of output labels to aggregate. Labels that are not 
          mentioned here are ignored. 
        - initial_weights provides a numeric weight to labelling sources.
          If left unspecified, the class assumes uniform weights.
          
        The class allows for multiple labelling to be valid for each text.
        Labels are incompatible with one another should be provided through the
        set_exclusive_labels method.
        """
        AbstractAggregator.__init__(self, name, labels)
        MajorityVoterMixin.__init__(self, initial_weights=initial_weights)
        MultilabelAggregatorMixin.__init__(self, NaiveBayes, initial_weights=initial_weights)


  
class MultilabelHMM(MultilabelAggregatorMixin, SequenceAggregatorMixin,
                                        AbstractAggregator):
    
    def __init__(self, name:str, labels:List[str], prefixes:str="BIO",
                 initial_weights:Optional[Dict[str,float]]=None):
        """Creates a new, multilabel aggregator for sequence labelling 
        using a HMM model. The parameters of the HMM model are learned without 
        access to the actual labels, using the Baum-Welch algorithm 
        (a special case of Expectation-Maximisation)
        
        Arguments:
        - name is the aggregator name
        - labels is a list of output labels to aggregate. Labels that are not 
          mentioned here are ignored. 
        - prefixes is the tagging scheme to use, such as IO, BIO or BILUO
        - initial_weights provides a numeric weight to labelling sources.
          If left unspecified, the class assumes uniform weights.
          
        The class allows for multiple labelling to be valid for each token.
        Labels are incompatible with one another should be provided through the
        set_exclusive_labels method.
        """
        AbstractAggregator.__init__(self, name, labels)
        SequenceAggregatorMixin.__init__(self, prefixes)
        MultilabelAggregatorMixin.__init__(self, HMM, initial_weights=initial_weights)
