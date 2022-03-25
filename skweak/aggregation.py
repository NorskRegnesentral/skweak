import pickle, copy, warnings
import numpy as np
from spacy.tokens import Doc, Span
from abc import abstractmethod
from typing import Iterable, List, Optional, Set, Dict, Tuple, Type

from .base import AbstractAnnotator
from spacy.tokens import Doc, Span  # type: ignore
from . import utils
import pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

####################################################################
# Abstract class for aggregators
####################################################################

class AbstractAggregator(AbstractAnnotator):
    
    def __init__(self, name: str, labels: List[str]):
        """Creates a new aggregator with the following arguments:
        - name is the aggregator name
        - labels is a list of output labels to aggregate, such as PERSON, ORG etc. 
          Labels that are not mentioned here are ignored. 
        """
        super(AbstractAggregator, self).__init__(name)
        if len(labels)==0:
            raise RuntimeError("Must have at least one label")
        self.out_labels = labels
        
        # We start by assuming that the list of labels produced by the 
        # labelling functions ("observed labels") and the output labels
        # after the aggregation are the same 
        self.observed_labels = list(labels)
        
        # we may also have label groups (for underspecified/overspecified labels)
        self.label_groups = {}

        
    def __call__(self, doc: Doc) -> Doc:
        """Aggregates all weak supervision sources"""

        # Extracting the sources to consider (and filtering out the ones to avoid)
        sources = self._get_sources_to_aggregate(doc)
        
        # If we have no labelling sources, we can skip the document
        if len(sources)==0:
            doc.spans[self.name] = []
            doc.spans[self.name].attrs = {"probs":{}, "aggregated":True, "sources":[]}
            return doc
            
        # Extracting the observation data
        df = self.get_observation_df(doc)
        
        # Special case: if the document has no observation whatsoever
        if len(df.columns)==0:
            output_spans, output_probs = [], []
            
        else: 
            # Filtering out rows with no relevant observations 
            df = self.filter_observations(df)
            
            # Running the actual aggregation
            agg_df = self.aggregate(df)  
                      
            # Convert the aggregate dataframe to spans
            output_spans = self._get_spans(agg_df)
            
            # And extract the full probability distributions
            output_probs = self._get_probs(agg_df)

        # Storing the results (both as spans and with the full probs)
        doc.spans[self.name] = [Span(doc, start, end, label=label)
                                for (start, end, label) in output_spans]
        doc.spans[self.name].attrs["probs"] = output_probs
        doc.spans[self.name].attrs["aggregated"] = True
        doc.spans[self.name].attrs["sources"] = list(df.columns)

        return doc
    
    
    def fit(self, docs: Iterable[Doc], **kwargs):
        """Fits the parameters of the aggregator model based on a collection
        of documents. The method extracts a dataframe of observations for
        each document and calls the _fit method"""
        
        obs_generator = (self.get_observation_df(doc) for doc in docs)
        self._fit(obs_generator, **kwargs)
        
        
    def _fit(self, all_obs:Iterable[pandas.DataFrame], **kwargs):
        """Fits the parameters of the aggregator model based on a collection
        of (span or token-level) observations extracted from documents. If 
        not overriden, the method assumes the model does not contain any 
        parameters, and does nothing"""    
        pass    
        

    def fit_and_aggregate(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        """Starts by fitting the parameters of the aggregator (if necessary)
        then applies the resulting model to aggregate the outputs of the
        labelling functions."""

        docs = list(docs)
        self.fit(docs)
        return list(self.pipe(docs))
    
        
    @abstractmethod
    def get_observation_df(self, doc: Doc):
        """Returns a dataframe containing the observed predictions of each labelling
        sources for the document. The dataframe has one row per unique span for span 
        labelling, and one row per token for sequence labelling."""

        raise NotImplementedError("must implement get_observation_df")    


    def filter_observations(self, obs:pandas.DataFrame) -> pandas.DataFrame:
        
        # We count the votes for each label on all sources 
        def count_fun(x):
            return np.bincount(x[x>=0], minlength=len(self.observed_labels)) 
        
        label_votes = np.apply_along_axis(count_fun, 1, obs.values).astype(np.float32)
        out_label_votes = label_votes.dot(self._get_vote_matrix())
        relevant_rows = (out_label_votes.sum(axis=1) > 0.0)
        
        return obs[relevant_rows] #type: ignore
        
        
    @abstractmethod
    def aggregate(self, obs: pandas.DataFrame) -> pandas.DataFrame:
        """Takes as input a 2D dataframe of shape (nb_entries, nb_sources) 
        associating each token/span to a set of observations from labelling 
        sources, and returns a 2D dataframe of shape (nb_entries, nb_labels)
        assocating each entry to the probability of each output label. 
        """

        raise NotImplementedError("must implement aggregate")        
    
    @abstractmethod
    def _get_spans(self, agg_df: pandas.DataFrame, threshold=0.5) \
        -> List[Tuple[int,int,str]]:
        """Takes as input a 2D dataframe of shape (nb_entries, nb_labels)
        assocating each token/span to the probability of an output label, and 
        returns a list of tuples (start, end, label).        
        """
        raise NotImplementedError("must implement _get_spans")    


    @abstractmethod
    def _get_probs(self, agg_df: pandas.DataFrame, min_threshold=0.1) \
        -> Dict[Tuple[int,int],Dict[str,float]]:
        """Takes as input a 2D dataframe of shape (nb_entries, nb_labels)
        assocating each token/span to the probability of an output label, and 
        returns a dictionary associating each (start,end) span/token to a 
        mapping from labels to probabilities"""

        raise NotImplementedError("must implement _get_probs")   
    
    
    def _get_sources_to_aggregate(self, doc):
        """Returns the sources to consider (and filtering out the ones to avoid, 
        such as aggretated sources)"""
        sources = [source for source in doc.spans if len(doc.spans[source]) > 0
                   and not doc.spans[source].attrs.get("aggregated", False)
                   and not doc.spans[source].attrs.get("avoid_in_aggregation", False)
                   and (not hasattr(self, "weights") 
                        or np.sum(self.weights.get(source, 1)) > 0) #type:ignore
                    and (not hasattr(self, "initial_weights") 
                        or self.initial_weights.get(source, 1) > 0)] #type:ignore
        return sources    


    def add_label_group(self, coarse_label: str, sub_labels: Set[str]):
        """Specifies that a "coarse" label is a placeholder for several possible
        fine-grained labels.  For instance, an ENT label may be a placeholder
        for multiple labels (PER, ORG, etc.)"""

        if coarse_label not in self.label_groups:
            self.label_groups[coarse_label] = set(sub_labels)
        else:
            self.label_groups[coarse_label].update(sub_labels)
            
        # Adding expansions of existing groups
        for other_coarse_label, other_sub_labels in self.label_groups.items():
            if other_coarse_label != coarse_label:
                for other_sub_label in list(other_sub_labels):
                    if other_sub_label==coarse_label:
                        other_sub_labels.update(sub_labels)
                        
        # And expansions based on existing groups
        for sub_label in list(sub_labels):
            if sub_label in self.label_groups:
                sub_labels.update(self.label_groups[sub_label])
                
        for label in sorted({coarse_label} | sub_labels):
            if label not in self.observed_labels:
                self.observed_labels.append(label)
                
    def add_underspecified_label(self, coarse_label: str, sub_labels: Set[str]):
        """Kept for backward compability purposes. The method add_label_group
        should be preferred."""
        
        self.add_label_group(coarse_label, sub_labels)
                
                
    def _get_vote_matrix(self, include_underspec=True) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a boolean matrice of shape (nb_observed_labels, nb_out_labels)
        which specifies possible mappings between observed labels and actual 
        output labels. 
        If include_underspec is set to True, we also include "partial" votes due to 
        underspecified labels (e.g. ENT)."""

        votes = np.zeros((len(self.observed_labels), 
                           len(self.out_labels)), dtype=np.float32)
        
        for i, label in enumerate(self.observed_labels):
            if label in self.out_labels:
                votes[i, self.out_labels.index(label)] = 1
        
        for coarse_label, fine_labels in self.label_groups.items():
            for fine_label in fine_labels:        
                        
                if include_underspec and fine_label in self.out_labels: # underspecification
                    coarse_label_ind = self.observed_labels.index(coarse_label)
                    fine_label_ind = self.out_labels.index(fine_label)
                    votes[coarse_label_ind, fine_label_ind] = 1.0/len(fine_labels)
                    
                elif coarse_label in self.out_labels: # overspecification
                    coarse_label_ind = self.out_labels.index(coarse_label)
                    fine_label_ind = self.observed_labels.index(fine_label)
                    votes[fine_label_ind, coarse_label_ind] = 1  
        
        return votes
    
        
    def save(self, filename):
        """Saves the aggregation model to a file"""
        fd = open(filename, "wb")
        pickle.dump(self, fd)
        fd.close()

    @classmethod
    def load(cls, pickle_file):
        """Loads the aggregation model from an existing file"""
        print("Loading", pickle_file)
        fd = open(pickle_file, "rb")
        ua = pickle.load(fd)
        fd.close()
        return ua



############################################
# Core aggregation tasks (text classification, sequence labelling, multi-labelling)
############################################   


class TextAggregatorMixin(AbstractAggregator):
    """Implementation of a subset of methods from AbstractAggregator when
    the aggregation is performed for text/span classification.
    This class should not be instantiated directly."""   

    def get_observation_df(self, doc: Doc):
        """Returns a dataframe containing the observed predictions of each labelling
        sources for the document. The content of the dataframe depends on the prefixes.
        The dataframe has one row per unique spans."""

        # Extracting the sources to consider (and filtering out the ones to avoid)
        sources = self._get_sources_to_aggregate(doc)

        # Extracts a list of unique spans (with identical boundaries)
        unique_spans = set((span.start, span.end)
                           for s in sources for span in doc.spans[s])
        sorted_spans = sorted(unique_spans)
        spans_indices = {span: i for i, span in enumerate(sorted_spans)}
        data = np.full((len(unique_spans), len(sources)),fill_value=-1, dtype=np.int16)

        observed_labels = self.observed_labels
        # Populating the array with the labels from each source
        label_indices = {l: i for i, l in enumerate(observed_labels)}
        
        for source_index, source in enumerate(sources):
            for span in doc.spans[source]:
                if span.label_ in observed_labels:
                    span_index = spans_indices[(span.start, span.end)]
                    data[span_index, source_index] = label_indices[span.label_]

        # We only consider spans with at least one concrete prediction
        masking = np.full(len(unique_spans), fill_value=True, dtype=bool)
        for i, row in enumerate(data):
            if row.max() < 0:
                masking[i] = False
        data = data[masking]
        sorted_unique_spans = [span for i, span in enumerate(sorted_spans) if masking[i]]

        # And we construct the final dataframe
        obs_df = pandas.DataFrame(data, columns=sources, index=sorted_unique_spans)
        return obs_df
        
    def _get_spans(self, agg_df: pandas.DataFrame, threshold=0.5) \
        -> List[Tuple[int,int,str]]:
        """Takes as input a 2D dataframe of shape (nb_entries, nb_labels)
        assocating each token/span to the probability of an output label, and 
        returns a list of tuples (start, end, label).        
        """
        results = []
        for (start,end), preds in agg_df.to_dict(orient="index").items():
            for label, prob in preds.items():
                if prob > threshold:
                    results.append((start,end,label))
   
        return results

    def _get_probs(self, agg_df: pandas.DataFrame, min_threshold=0.1) \
        -> Dict[Tuple[int,int],Dict[str,float]]:
        """Takes as input a 2D dataframe of shape (nb_entries, nb_labels)
        assocating each span to the probability of an output label, and returns
        a dictionary associating each (start,end) span/token to a mapping
        from labels to probabilities"""

        return {span: {label: prob for label, prob in distrib.items() 
                       if prob >= min_threshold} #type: ignore
                for span, distrib in agg_df.to_dict(orient="index").items()}
    
    
        
class SequenceAggregatorMixin(AbstractAggregator):
    """Implementation of a subset of methods from AbstractAggregator when
    the aggregation is performed for token-level sequence labelling.
    This class should not be instantiated directly"""

    def __init__(self, prefixes: str = "BIO"):
        """Do not call this initializer directly, and use the fully
        implemented classes (MajorityVoter, NaiveBayes, HMM, etc.) instead"""

        if prefixes not in {"IO", "BIO", "BILUO", "BILOU"}:
            raise RuntimeError(
                "Tagging scheme must be 'IO', 'BIO', 'BILUO' or ''")
        
        # We add prefixes to the labels (i.e. "B-ORG" etc)
        prefixed_labels = ["O"]
        for label in self.out_labels:
            if any(label.startswith("%s-"%p) for p in "BILU"):
                raise RuntimeError("Initial labels should not have a prefix")
            for prefix in prefixes.replace("O", ""):
                prefixed_labels.append("%s-%s" % (prefix, label))
        
        self.prefixes = prefixes    
        self.out_labels_no_prefixes=list(self.out_labels)

        self.out_labels = prefixed_labels
        self.observed_labels = list(prefixed_labels)

        

    def _get_spans(self, agg_df: pandas.DataFrame) -> List[Tuple[int,int,str]]:
        """Takes as input a 2D dataframe of shape (nb_entries, nb_labels)
        assocating each token to the probability of an output label, and 
        returns a list of tuples (start, end, label).        
        """
        return [(start, end, label) for (start, end), label in 
                utils.token_array_to_spans(agg_df.values, self.out_labels).items()]

    @abstractmethod
    def _get_probs(self, agg_df: pandas.DataFrame) -> Dict:
        """Takes as input a 2D dataframe of shape (nb_entries, nb_labels)
        assocating each token to the probability of an output label, and returns
        a dictionary mapping each token to a distribution over labels"""

        return utils.token_array_to_probs(agg_df.values, self.out_labels)
    

    def get_observation_df(self, doc: Doc):
        """Returns a dataframe containing the observed predictions of each labelling
        sources for the document. The content of the dataframe depends on the prefixes.
        The dataframe has one row per token."""

        # Extracting the sources to consider (and filtering out the ones to avoid)
        sources = self._get_sources_to_aggregate(doc)

        data = utils.spans_to_array(doc, self.observed_labels, sources)
        return pandas.DataFrame(data, columns=sources)


    def add_label_group(self, coarse_label: str, sub_labels: Set[str]):
        """Specifies that a "coarse" label is a placeholder for several possible
        fine-grained labels.  For instance, an ENT label may be a placeholder
        for multiple labels (PER, ORG, etc.)"""
        
        if coarse_label in self.out_labels and all(l in self.observed_labels for l in sub_labels):
            super().add_label_group(coarse_label, sub_labels)
            return
        
        for prefix in self.prefixes:
            if prefix !="O":
                coarse_label_with_prefix = "%s-%s"%(prefix, coarse_label)
                sub_labels_with_prefixes = {"%s-%s"%(prefix, sub_label) 
                                            for sub_label in sub_labels if sub_label!="O"}
                super().add_label_group(coarse_label_with_prefix, sub_labels_with_prefixes)
                     

class MultilabelAggregatorMixin(AbstractAggregator):
    """Functionalities for multilabel classification or sequence labelling. This
    class should not be instantiated directly"""
    
    def __init__(self, aggregator:Type[AbstractAggregator], **kwargs):
        
        self.models = {}
        
        # If the underlying model is for sequence labelling, we create one
        # separate model for detecting each entity
        if isinstance(self, SequenceAggregatorMixin):
            for label in self.out_labels_no_prefixes: #type: ignore
                args = {"name":self.name + "_base", "labels":[label], "prefixes":self.prefixes, **kwargs}
                self.models[label] = aggregator(**args)
                self.models[label].observed_labels = list(self.observed_labels)
        else:
            for label in self.out_labels: #type: ignore
                args = {"name":self.name + "_base", "labels":[f"NOT/{label}", label], **kwargs}
                self.models[label] = aggregator(**args)
                self.models[label].observed_labels = list(self.observed_labels)
       
           
    def set_exclusive_labels(self, exclusive_labels:Set[str]):
        """Defines a set of labels that are mutually exclusive (i.e. they cannot 
        simultaneously co-occur for the same data point)"""
        
        for label, model in self.models.items():
            if label in exclusive_labels:
                other_labels = exclusive_labels - {label}
                if hasattr(self, "prefixes"):
                    other_labels = {f"{p}-{other_label}" for p in self.prefixes if p!="O" #type: ignore
                                    for other_label in other_labels}
                    model.add_label_group("O", other_labels)
                else:
                    model.add_label_group(f"NOT/{label}", other_labels)
                
    def add_label_group(self, coarse_label:str, sub_labels:Set[str]):
        """Specifies that a "coarse" label is a placeholder for several possible
        fine-grained labels.  For instance, an ENT label may be a placeholder
        for multiple labels (PER, ORG, etc.)"""
        
        for model in self.models.values():
            model.add_label_group(coarse_label, sub_labels)
    
    def aggregate(self, obs: pandas.DataFrame) -> pandas.DataFrame:
        """Takes as input a 2D dataframe of shape (nb_entries, nb_sources) 
        associating each token/span to a set of observations from labelling 
        sources, and returns a 2D dataframe of shape (nb_entries, nb_labels)
        assocating each entry to the probability of each output label. 
        
        Note: for multilabelling, there may of course be several labels that 
        are simulteanously true for a given data point.
        """
        
        results = {}
        
        for label, sub_model in self.models.items(): #type: ignore
            
            # We filter out rows with no relevant observations        
            relevant_obs = sub_model.filter_observations(obs)
            
            # We run the aggregation on each model separately
            result = sub_model.aggregate(relevant_obs)
            # We can discard the "O" or NOT/label probabilities
            results[label] = result[sub_model.out_labels[1:]]
            
        # We ensure the probabilities over exclusive labels are normalised
        norm_results = {}
        for label, probs in results.items():
            prob_sum = probs.copy().sum(axis=1)
            if hasattr(self, "prefixes"):
                exclusives = self.models[label].label_groups.get("O", []) #type: ignore
                exclusives = {exclusive.split("-")[-1] for exclusive in exclusives}
            else:
                exclusives = self.models[label].label_groups.get(f"NOT/{label}", []) #type: ignore
            for exclusive_label in exclusives:
                prob_sum += results[exclusive_label].sum(axis=1)
            normalisation = np.where(prob_sum > 1.0, prob_sum, 1.0)
            norm_results[label] = probs / (normalisation[:,np.newaxis]) # type:ignore
            
        # We concatenate all results, and fill unknown probabilities with 0
        results = pandas.concat(norm_results.values(), axis=1)
        results = results.fillna(0.0)

        if hasattr(self, "prefixes"):
            residual_prob = np.prod(1-results.values, axis=1)
            results.insert(loc=0, column='O', value=residual_prob)
        
        return results
        
        
    def _fit(self, all_obs:Iterable[pandas.DataFrame]):
        """We fit the set of aggregation models based on the observations"""

        # We store all observations
        if not isinstance(all_obs, list):
            all_obs = list(all_obs)

        # And then fit each model individually
        for sub_model in self.models.values(): #type: ignore
            sub_model._fit(all_obs)


from . import generative, voting

def MajorityVoter(name: str, labels: List[str], sequence_labelling: bool = True,
                 initial_weights=None, prefixes: str = "BIO"):
    """Added for backward compability purposes. See module voting for updated classes"""
    if sequence_labelling:
        return voting.SequentialMajorityVoter(name, labels, prefixes=prefixes,
                                              initial_weights=initial_weights)
    else:
        return voting.MajorityVoter(name, labels, initial_weights=initial_weights)
    
def HMM(name: str, out_labels: List[str], sequence_labelling: bool = True,
        prefixes: str = "BIO",  initial_weights=None, redundancy_factor=0.1):
    """Added for backward compability purposes. See module generative for updated classes"""
    
    if sequence_labelling:
        return generative.HMM(name, out_labels, prefixes=prefixes, 
                              initial_weights=initial_weights, 
                              redundancy_factor=redundancy_factor)
    else:
        return generative.NaiveBayes(name, out_labels, initial_weights=initial_weights, 
                                     redundancy_factor=redundancy_factor)