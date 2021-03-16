from __future__ import annotations

from abc import abstractmethod
from typing import Dict, Iterable, List, Set
import numpy as np 
from .base import BaseAnnotator
from spacy.tokens import Doc
from . import utils
import pickle, math, itertools
import hmmlearn
import hmmlearn.base

####################################################################
# Aggregation models
####################################################################


class BaseAggregator(BaseAnnotator):
    """Base class for all types of aggregator components"""
    
    def __init__(self, name:str, out_labels:List[str], encoding="BIO"):
        """Creates a new aggregator with the provided name. out_labels must
        be a list of labels such as PERSON, ORG etc. sources must be a list of labelling
        sources to aggregate."""
        
        super(BaseAggregator, self).__init__(name)
        
        # Checking the encoding scheme
        if encoding not in {"IO", "BIO", "BILUO", "BILOU"}:
            raise RuntimeError("Tagging scheme must be IO, BIO or BILUO")
        self.encoding = encoding
        
        self.out_labels = out_labels
        self.sources_to_avoid = []

        # We may also have "constraint labels" that may stand for several
        # possible output labels (see below) 
        self.constraint_labels = {} 


    def add_constraint_label(self, label: str, satisfied_values: Set[str]):
        """Specifies that the given label is not a specific output but a constraint
        that may be satisfied by several values. For instance, a source could label 
        a span as ENT (indicating that the span can receive any non-null label) 
        or NOT-PERSON (indicating that the label should not be a person)."""
        
        self.constraint_labels[label] = satisfied_values
       
    def __call__(self, doc: Doc) -> Doc:
        """Aggregates all weak supervision sources"""
        
        if "spans" in doc.user_data:        
                        
            sources = {s for s in doc.user_data["spans"] if not 
                       any((to_avoid in s) for to_avoid in self.sources_to_avoid)}
            if not sources:
                return doc
            source_arrays = utils.spans_to_arrays(doc, self.observed_labels, sources, self.encoding)

            # Running the actual aggregation
            agg_array = self._aggregate(source_arrays)
            
            # Converting back to spans or token labels
            spans = utils.array_to_spans(agg_array, self.prefix_out_labels)
            token_labels = utils.array_to_token_labels(agg_array, self.prefix_out_labels)
            
            # Storing the results (both as spans and as token labels)
            if "agg_spans" not in doc.user_data:
                doc.user_data["agg_spans"] = {self.name: spans}
                doc.user_data["agg_token_labels"] = {self.name: token_labels}
            else:
                doc.user_data["agg_spans"][self.name] = spans
                doc.user_data["agg_token_labels"][self.name] = token_labels
        
        return doc
            
            
    @abstractmethod
    def _aggregate(self, label_arrays: Dict[str,np.ndarray]) -> np.ndarray:
        """Takes as input a dictionary associating each labelling source with a
        2D boolean array of shape (nb_tokens, nb_prefix_labels) and returns a 
        2D array (nb_tokens, nb_prefix_out_labels) with float values 
        expressing the probabilities of each possible label for a given token.
        
        Note that the 2D input arrays for each source will include both concrete 
        output labels as well as constraint labels (such as NOT-PERSON), while 
        the 2D output array will only contain concrete output labels. 
        """
        
        raise NotImplementedError("must implement aggregate_spans")

    
    @property
    def prefix_out_labels(self) -> List[str]:
        """List of output labels in full prefix form (B-PERSON etc.)"""
    
        return (["O"] + ["%s-%s"%(prefix, l) for l in self.out_labels 
                         for prefix in self.encoding.replace("O", "")])
    

    @property
    def observed_labels(self) -> List[str]:
        """Returns the possible labels that can be observed in labelling sources
        (that is, either "concrete" output labels or constraint labels)."""
        
        return self.out_labels + list(self.constraint_labels.keys())
    
    @property
    def prefix_observed_labels(self) -> List[str]:
        """List of possible observed labels (both concrete output labels and 
        constraint labels) in prefix form"""
        
        return (self.prefix_out_labels +  
                ["%s-%s"%(prefix, l) for l in self.constraint_labels 
                 for prefix in self.encoding.replace("O", "")])
        
        
    def get_constraint_satisfaction_matrix(self) -> np.ndarray: 
        """Creates a boolean matrix of shape (nb_prefix_constraint_labels, nb_out_labels)
        which specifies, for each constraint label in prefix form (like B-ENT),
        the set of concrete output labels satisfying it (like B-ORG, B-PERSON,etc.)"""
        
        constraint_satisfaction_rows = []
        for constraint_label, satisfied_values in self.constraint_labels.items():
            for prefix in self.encoding.replace("O", ""):      
                row = [label[0]==prefix and label[2:] in satisfied_values 
                       for label in self.prefix_out_labels]
                constraint_satisfaction_rows.append(row)
        return np.array(constraint_satisfaction_rows)
         
 
    
                 

class MajorityVoter(BaseAggregator):
    """Simple aggregator based on majority voting"""
    
    def __init__(self, name: str, out_labels: List[str], min_nb_votes=1):
        """Creates a majority voter to aggregate spans and document-level 
        classes. 

        min_nb_votes corresponds to the minimum number of sources assigning a
        non-"O" label in order to start voting. Otherwise, we assume a "O" label.
        """
        
        super(MajorityVoter, self).__init__(name, out_labels)
        self.min_nb_votes = min_nb_votes
        self.weights = None 
    
          
    def _aggregate(self, input_arrays: Dict[str,np.ndarray]) -> np.ndarray:
        """Takes as input a dictionary associating each labelling source with a
        2D boolean array of shape (nb_tokens, nb_prefix_labels) and returns a 
        2D array (nb_tokens, nb_prefix_out_labels) with float values 
        expressing the probabilities of each possible label for a given token.
                
        This probability is here computed based on making each source "vote"
        on its output label. The most likely label will thus be the one that
        is indicated by most sources. If constraint labels are included, they 
        are also part of the vote count. """
        
        # We count the votes for each label on all sources
        label_votes = np.add.reduce([arr.astype(np.int32) 
                                     for arr in input_arrays.values()])  
        if self.weights:
            source_weights = [self.weights.get(s, 1.0) for s in input_arrays]
            label_votes = np.multiply(label_votes, source_weights)
        
        # We check the rows including a minimum number of votes (on concrete labels)
        concrete_label_votes = label_votes[:,1:len(self.prefix_out_labels)]
        rows_with_enough_votes = concrete_label_votes.sum(axis=1) >= self.min_nb_votes
        concrete_label_votes = concrete_label_votes[rows_with_enough_votes]

        # We also add to the votes the values satisfying constraints
        if self.constraint_labels:
            constraint_label_votes = label_votes[rows_with_enough_votes,len(self.prefix_out_labels):]
            constraint_satisfaction_matrix = self.get_constraint_satisfaction_matrix()[:,1:]
            additional_votes = constraint_label_votes.dot(constraint_satisfaction_matrix)  
            concrete_label_votes += (additional_votes * concrete_label_votes.astype(bool))
        
        
        # We define the output label array
        # And fill in with the votes (after normalisation)
        total = np.expand_dims(concrete_label_votes.sum(axis=1), axis=1)
        concrete_label_votes = concrete_label_votes / total
            
        output_array = np.zeros((len(label_votes), len(self.prefix_out_labels)))
        output_array[rows_with_enough_votes, 1:len(self.prefix_out_labels)] = concrete_label_votes
    
        # Tokens without predictions should get a "O" label                
        output_array[:,0] = 1-output_array.sum(axis=1)
        return output_array
    
    
class EnsembleMajorityVoter(BaseAggregator):
    """Ensemble of majority voters, each with a distinct minimal threshold of 
    non-'O' labels to trigger the majority voter. """
    
    def _aggregate(self, input_arrays: Dict[str,np.ndarray]) -> np.ndarray:
        
        # We compute an aggregation of the input arrays for each voter
        agg_array = 0
        for min_nb_votes in range(1, len(input_arrays)+1):
            mv = MajorityVoter("",self.out_labels, min_nb_votes=min_nb_votes)
            mv.constraint_labels = self.constraint_labels
            mv.sources_to_avoid = self.sources_to_avoid
            mv.weights = getattr(self, "weights", {})

            agg_array_for_mv = mv._aggregate(input_arrays) 

            # We stop once the number of non-O tokens is less than 1/3 of 
            # the number with min_nb_votes=1
            nb_non_null_tokens = len(agg_array_for_mv)- agg_array_for_mv[:,0].sum()
            if min_nb_votes==1:
                nb_non_null_tokens_one_vote = nb_non_null_tokens
            elif nb_non_null_tokens < nb_non_null_tokens_one_vote / 3:
                break
            agg_array += agg_array_for_mv
     
        total = np.expand_dims(agg_array.sum(axis=1), axis=1)
        agg_array = agg_array / total

        return agg_array
            
                  

    
class HMM(hmmlearn.base._BaseHMM, BaseAggregator):
    """Aggregator for labelled spans based on a HMM model. The HMM model is learned
    without access to the actual labels, using the Baum-Welch algorithm 
    (a special case of Expectation-Maximisation)"""
  
    def __init__(self, name:str, out_labels:List[str], encoding="BIO", add_dependencies=True):
        """Initialises the HMM model (which must be fitted before being used)"""
        
        BaseAggregator.__init__(self, name, out_labels, encoding)
        self.add_dependencies = add_dependencies
    

    def _aggregate(self, input_arrays: np.ndarray) -> np.ndarray :
        """Makes a list of predicted labels (using Viterbi) for each token, along with
        the associated probability according to the HMM model."""
        
        if not hasattr(self, "emit_probs"):
            raise RuntimeError("Model is not yet trained")
            
        # Compute the log likelihoods for each states
        framelogprob = self._compute_log_likelihood(input_arrays)

        # Run a forward pass
        _, forward_lattice = self._do_forward_pass(framelogprob)     
        forward_lattice = forward_lattice - forward_lattice.max(axis=1)[:,np.newaxis]
        
        # Transform into probabilities
        posteriors = np.exp(forward_lattice) 
        posteriors = posteriors / posteriors.sum(axis=1)[:,np.newaxis]
               
        return posteriors


    def fit(self, docbin_file:str, cutoff:int=None, n_iter=10, tol=1e-2):
        """Train the HMM annotator based on the docbin file"""

        # We extract the docs from the file
        docs = utils.docbin_reader(docbin_file, cutoff=cutoff)
        
        # We extract all source names 
        sources, docs = self._extract_sources(docs)

        # And add correlations between them
        self._add_correlations(sources)

        # Create uninformed priors to start with
        self._reset_counts(sources, with_priors=True)

        # And add the counts from an ensemble of majority voters
        self._add_emv_counts(docs)

        # Finally, we postprocess the counts and get probabilities
        self._do_mstep()

        monitor = hmmlearn.base.ConvergenceMonitor(tol, n_iter, True)
        monitor._reset()
        for iter in range(n_iter):
            print("Starting iteration", (iter+1))
            curr_logprob = 0
            self._reset_counts(sources)
            nb_docs = 0

            # We loop on all documents at each iteration
            for doc in utils.docbin_reader(docbin_file, cutoff=cutoff):
                
                # Transform the document annotations into boolean arrays
                X = utils.spans_to_arrays(doc, self.observed_labels, sources, self.encoding) 

                # Compute its current log-likelihood
                framelogprob = self._compute_log_likelihood(X)

                # Make sure there is no token with no possible states
                if framelogprob.max(axis=1).min() < -100000:  
                    pos = framelogprob.max(axis=1).argmin() 
                    print("problem found for token", doc[pos])
                    return framelogprob

                # We run a forward and backward pass to compute the posteriors
                logprob, fwdlattice = self._do_forward_pass(framelogprob)
                curr_logprob += logprob
                bwdlattice = self._do_backward_pass(framelogprob)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
                
                # We accumulate the statistics in the counts
                self._accumulate_statistics(X, framelogprob, posteriors, fwdlattice, bwdlattice)
                nb_docs += 1
                
                if nb_docs % 1000 == 0:
                    print("Number of processed documents:", nb_docs)
            print("Finished E-step with %i documents"%nb_docs)
            
            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            self._do_mstep()

            monitor.report(curr_logprob)
            if monitor.converged:
                break
        return self

    def _extract_sources(self, docs:Iterable[Doc], max_number=100):
        """Extract the names of all labelling sources mentioned in the documents
        (and not included in the list of sources to avoid)"""
        sources = set()
        sampled_docs = []
        for i, doc in enumerate(docs):
            for source in doc.user_data.get("spans", {}):
                if source not in sources and source not in self.sources_to_avoid:
                    sources.add(source)
            sampled_docs.append(doc)
            if i > max_number:
                break
        
        # We do not want to "consume" docs for this operation, so we add them back
        docs = itertools.chain(sampled_docs, docs)
        return sources, docs


    def _add_correlations(self, sources):
        """ Look for dependencies between labelling sources according to specific conventions:
        a) If two source names are respectively written as "someprefix_sourcename" and 
        "sourcename", then we assume that "some_prefix_sourcename" is a specialisation of 
        "sourcename" and is therefore correlated with it. 
        b) Similarly, if a source ends with "_cased", and there is an "_uncased" version of 
        the same source, we add a correlation from the cased to the uncased version. 
        NB: Those conventions can of course be easily adapted to your particular case. """
       
        self.dependencies = {}
        self.corr_weights = {}
        if not self.add_dependencies:
            return            

        for source in sources:
            if "_" in source and source.split("_",1)[1] in sources:
                self.dependencies[source] = source.split("_",1)[1]
            elif "_cased" in source and source.replace("_cased", "_uncased") in sources:
                self.dependencies[source] = source.replace("_cased", "_uncased")
                
 
    def _reset_counts(self, sources, with_priors=False):
        """Reset the various counts/statistics used for for the M-step. If the with_priors
        flag is set, also adds uninformed priors for the start, transition and emission counts"""
  
        nb_labels = len(self.prefix_out_labels)
        nb_obs = len(self.prefix_observed_labels)
        
        # We reset all counts
        self.start_counts = np.zeros(shape=(nb_labels,))
        self.trans_counts = np.zeros(shape=(nb_labels, nb_labels))
        self.emit_counts = {source:np.zeros(shape=(nb_labels, nb_obs)) for source in sources}
        self.corr_counts = {source:np.zeros(shape=(nb_obs, nb_obs)) for source in self.dependencies}

        # Square difference between predicted and actual observations for the emission model
        # (based on the state) and the model based on observations from a correlated 
        # source). Those are used to determine the relative weight of the two models
        self.emit_diff = {source:np.zeros(nb_obs) for source in self.dependencies}
        self.corr_diff = {source:np.zeros(nb_obs) for source in self.dependencies}

        # We add some prior values
        if with_priors:
            self.start_counts += 1.000001
            self.trans_counts += 1.000001
            for source in sources:
                self.emit_counts[source][:,:nb_labels] = np.eye(nb_labels)
                self.emit_counts[source][:,0] = 1
                self.emit_counts[source][0,:] = 1
                self.emit_counts[source] += 0.01
            
            for source in self.dependencies:
                self.corr_counts[source] = np.eye(nb_obs)
                self.corr_counts[source][:,0] = 1
                self.corr_counts[source][0,:] = 1
                self.corr_counts[source] += 0.01
                self.emit_diff[source] += 0.01
                self.corr_diff[source] += 0.01
        

    def _add_emv_counts(self, docs:Iterable[Doc]):
        """Getting initial counts for the HMM parameters based on an ensemble of
        majority voters"""
       
        # We rely on an ensemble majority voter to get the first counts
        emv = EnsembleMajorityVoter("",self.out_labels)
        emv.constraint_labels = self.constraint_labels
        emv.sources_to_avoid = self.sources_to_avoid

        # We give lower weights to labelling sources that are correlated         
        # emv.weights = {source:0.5 for source in self.dependencies.values()}
 
        for doc in docs:
                  
            # We create a boolean array of emissions for each source   
            input_arrays = utils.spans_to_arrays(doc, self.observed_labels, 
                                                 set(self.emit_counts.keys()), self.encoding)
            
            # And aggregate the results
            agg_array = emv._aggregate(input_arrays)
            
            # Update the start probabilities
            self.start_counts += agg_array[0, :] 
            
            # Update the transition probabilities
            for i in range(1, len(agg_array)):
                self.trans_counts += np.outer(agg_array[i-1], agg_array[i]) 
            
            # Update the emission probabilities
            for source in self.emit_counts:
                mv_counts = np.dot(agg_array.T, input_arrays[source])    
                self.emit_counts[source] += mv_counts
                
            for source in self.corr_counts:
                dep = input_arrays[self.dependencies[source]]
                self.corr_counts[source] += np.dot(dep.T, input_arrays[source])

        

 
    def _do_mstep(self):
        """Performs the maximisation step of the EM algorithm"""
        
        # We do some postprocessing of the counts to erase invalid counts
        self._postprocess_counts()

        # We normalise to get probabilities
        self.startprob_ = self.start_counts / (self.start_counts.sum() + 1E-100)    

        trans_norm =  (self.trans_counts.sum(axis=1) + 1E-100)[:,np.newaxis]
        self.transmat_ = self.trans_counts / trans_norm

        self.emit_probs = {}
        for source in self.emit_counts:
            normalisation = (self.emit_counts[source] + 1E-100).sum(axis=-1)[:,np.newaxis]
            self.emit_probs[source] = self.emit_counts[source] / normalisation

        self.corr_probs = {}
        for source in self.corr_counts:
            normalisation = (self.corr_counts[source] + 1E-100).sum(axis=-1)[:,np.newaxis]
            self.corr_probs[source] = self.corr_counts[source] / normalisation
    
            # For observations that are conditionally dependent on both the state and another
            # source, computes the relative weight of the two based on the square difference
            # between the predicted and actual observations for the two models. The lower the
            # square difference (compared to the other model), the higher the weight. 
            diff_norm = self.corr_diff[source] + self.emit_diff[source] + 0.0002 
            self.corr_weights[source] = (1 - (self.corr_diff[source] + 0.0001) / diff_norm)
            


    def _postprocess_counts(self):
        """Postprocess the counts to erase invalid starts, transitions or emissions"""

        # We make sure the counts for invalid starts (i.e. "L-ORG") are zero
        for i, label in enumerate(self.prefix_out_labels):
            if not utils.is_valid_start(label, self.encoding):
                self.start_counts[i] = 0

        # We make sure the counts for invalid transitions (i.e. B-ORG -> I-GPE) are zero
        for i, label in enumerate(self.prefix_out_labels):
            for j, label2 in enumerate(self.prefix_out_labels):
                if not utils.is_valid_transition(label, label2, self.encoding):
                    self.trans_counts[i,j] = 0
                    
        # We also take into account the constraint matrix (but in a soft manner)
        for emit_counts in self.emit_counts.values():
            cur_counts = emit_counts[:,len(self.prefix_out_labels):]
            constraints = self.get_constraint_satisfaction_matrix().T
            new_counts = 0.1 * cur_counts + 0.9 * cur_counts * constraints
            emit_counts[:,len(self.prefix_out_labels):] = new_counts



    def _compute_log_likelihood(self, X):
        """Computes the log likelihood for the observed sequence"""
        logsum = None
        for source in sorted(self.emit_counts.keys()):
            if source in X:
                # We compute the likelihood of each state given the source labels
                probs = np.dot(X[source], self.emit_probs[source].T) 

                # For correlated sources, we also take the other source into account
                if source in self.dependencies:
                    dep = self.dependencies[source]
                    weights = np.dot(X[dep], self.corr_weights[source])[:,np.newaxis]
                    corr_probs = np.dot(X[dep], self.corr_probs[source])[X[source]]
                    probs = (1-weights)*probs + weights*corr_probs[:,np.newaxis]

                # Impossible states have a logprob of -inf
                probs = np.ma.log(probs).filled(-np.inf)
                logsum = probs if logsum is None else (logsum + probs)

        # We also add a constraint that the probability of a state is zero 
        # if no labelling functions observes it
        X_all_obs = np.zeros(logsum.shape, dtype=bool) 
        for source in self.emit_counts:
            if source in X:
                X_all_obs += X[source][:,:len(self.prefix_out_labels)]
        logsum = np.where(X_all_obs, logsum, -np.inf)
        return logsum


    def _accumulate_statistics(self, X, framelogprob, posteriors, fwdlattice, bwdlattice):
        """Acccumulate the counts based on the sufficient statistics"""

        # Update the start counts
        self.start_counts += posteriors[0]

        # Updating the transition counts
        n_samples, n_components = framelogprob.shape
        if n_samples > 1:
            log_xi_sum = np.full((n_components, n_components), -np.inf)
            hmmlearn._hmmc._compute_log_xi_sum(n_samples, n_components, fwdlattice,
                                    hmmlearn.base.log_mask_zero(self.transmat_),
                                    bwdlattice, framelogprob, log_xi_sum)
            self.trans_counts += np.exp(log_xi_sum)

        # Updating the emission counts
        for source in self.emit_counts:
            self.emit_counts[source] += np.dot(posteriors.T, X[source])

        # Updating the emission counts
        for source, dep in self.dependencies.items():
            self.corr_counts[source] += np.dot(X[dep].T, X[source])

        # Update the square differences for the emissions and conditional models
        for source, dep in self.dependencies.items():
            emit_square_diff = ((np.dot(posteriors, self.emit_probs[source]) - X[source])**2).sum(axis=1)
            corr_square_diff = ((np.dot(X[dep], self.corr_probs[source])- X[source])**2).sum(axis=1)
            self.emit_diff[source] += np.dot(X[dep].T, emit_square_diff)
            self.corr_diff[source] += np.dot(X[dep].T, corr_square_diff)



    def pretty_print(self, sources=None, nb_digits=2):
        """Prints out a summary of the HMM models"""
        
        import pandas
        pandas.set_option("display.width", 1000)
        print("HMM model on following sources:", list(self.emit_counts.keys()))
        print("Output labels:", self.out_labels)
        if self.constraint_labels:
            print("Constraint labels:", self.constraint_labels)
        print("--------")
        print("Start distribution:")
        print(pandas.Series(self.startprob_, index=self.prefix_out_labels).round(nb_digits))
        print("--------")
        print("Transition model:")
        print(pandas.DataFrame(self.transmat_, index=self.prefix_out_labels, 
                               columns=self.prefix_out_labels).round(nb_digits))
        print("--------")
        for source in self.emit_counts:
            if sources == None or source in sources:
                print("Emission model for source:", source)
                df = pandas.DataFrame(self.emit_probs[source], index=self.prefix_out_labels,  
                                    columns=self.prefix_observed_labels)
                print(df.round(nb_digits))
                print("--------")
        for source in self.corr_counts:
            if sources == None or source in sources:
                print("Correlation model for source: %s (dependent: %s)"%(source, self.dependencies[source]))
                df = pandas.DataFrame(self.corr_probs[source], index=self.prefix_observed_labels,  
                                    columns=self.prefix_observed_labels).round(nb_digits)
                df["weight"] = self.corr_weights[source]
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
    
# class SnorkelAggregator(BaseAggregator):
#     """Snorkel-based model. The model first extracts a list of candidate spans 
#     from a few trustworthy sources, and then relies on the full set of sources
#     for the classification"""
    
#     def __init__(self, name:str, out_labels:List[str], sources:List[str]):
#         super(SnorkelAggregator, self).__init__(name, out_labels, sources)
#         self.sources = sources
        
#     def train(self, docbin_file):
#         """Trains the Snorkel model on the provided corpus"""
        
#         import snorkel.labeling
#         all_obs = []
#         for doc in utils.docbin_reader(docbin_file):
#             spans, obs = self._get_inputs(doc)
#             all_obs.append(obs)
#             if len(all_obs) > 5:
#                 break
#         all_obs = np.vstack(all_obs)
#         self.label_model = snorkel.labeling.LabelModel(len(self.out_labels) + 1)
#         self.label_model.fit(all_obs)
        
        
#     def _get_inputs(self, doc):
#         """Returns the list of spans and the associated labels for each source (-1 to abtain)"""
        
#         spans = sorted(utils.get_spans(doc, self.sources))
#         span_indices = {span:i for i, span in enumerate(spans)}
#         obs = np.full((len(spans), len(self.sources)+1), -1)
        
#         label_map = {label:i for i, label in enumerate(self.out_labels)}
        
#         for source_index, source in enumerate(self.sources):
#             if source in doc.user_data["spans"]:
#                 for (start,end), label in doc.user_data["spans"][source].items():
#                     if (start,end) in span_indices:
#                         span_index = span_indices[(start,end)]
#                         obs[span_index, source_index] = label_map[label]
                        
#         return spans, obs
        
        
#     def annotate(self, doc):
#         """Annotates the document with the Snorkel output"""
        
#         doc.user_data["annotations"][self.source_name] = {}
#         doc = self.specialise_annotations(doc)
#         spans, obs = self._get_inputs(doc)
#         predict_probs = self.label_model.predict_proba(obs)
#         for (start,end), probs_for_span in zip(spans, predict_probs):
#             label_index = probs_for_span.argmax()
#             if label_index > 0:
#                 label = LABELS[label_index-1]
#                 prob = probs_for_span.max()
#                 doc.user_data["annotations"][self.source_name][(start,end)] = ((label, prob),)
#         return doc
    