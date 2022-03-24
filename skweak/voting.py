
import pickle, copy, warnings
import numpy as np
from spacy.tokens import Doc, Span
from abc import abstractmethod
from typing import Iterable, List, Optional, Set, Dict, Tuple, Type

from .aggregation import AbstractAggregator,TextAggregatorMixin,SequenceAggregatorMixin,MultilabelAggregatorMixin
from spacy.tokens import Doc, Span  # type: ignore
from . import utils
import pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

    
############################################
# Majority voting
############################################   


class MajorityVoterMixin(AbstractAggregator):
    """Implementation of a subset of methods from AbstractAggregator when
    the aggregation is performed for text/span classification.
    This class should not be instantiated directly."""
     
    def __init__(self, initial_weights=None):
        """Do not call this initializer directly, and use the fully
        implemented classes (MajorityVoter, NaiveBayes, HMM, etc.) instead"""
           
        # initial_weights is a dictionary associating source names to numerical weights
        #  in the range [0, +inf]. The default assumes weights = 1 for all functions. You
        #  can disable a labelling function by giving it a weight of 0.        """

        self.weights = initial_weights if initial_weights else {}

    def aggregate(self, obs: pandas.DataFrame) -> pandas.DataFrame:
        """Takes as input a 2D dataframe of shape (nb_entries, nb_sources) 
        associating each token/span to a set of observations from labelling 
        sources, and returns a 2D dataframe of shape (nb_entries, nb_labels)
        assocating each entry to the probability of each output label. 

        This probability is here computed based on making each source "vote"
        on its output label. The most likely label will thus be the one that
        is indicated by most sources. If underspecified labels are included, they 
        are also part of the vote count. """

        weights = np.array([self.weights.get(source, 1) for source in obs.columns])

        # We count the votes for each label on all sources 
        # (taking weights into account)
        def count_fun(x):
            return np.bincount(x[x>=0], weights=weights[x>=0], 
                               minlength=len(self.observed_labels)) 
        label_votes = np.apply_along_axis(count_fun, 1, obs.values).astype(np.float32)
        
        # For token-level sequence labelling, we need to normalise the number 
        # of "O" occurrences, since they both indicate the absence of 
        # prediction, but are also a possible output
        if self.observed_labels[0]=="O":
            label_votes = self.normalise_o_labels(label_votes)

        # We transform the votes from observations into output labels,
        out_label_votes = label_votes.dot(self._get_vote_matrix())

        # Normalisation
        total = np.expand_dims(out_label_votes.sum(axis=1), axis=1)
        probs = out_label_votes / (total + 1E-30)
        df = pandas.DataFrame(probs, index=obs.index, columns=self.out_labels)
        return df
      
      
    def normalise_o_labels(self, label_votes, power_base=3.0):
        """The normalised counts for the O labels are defined as B^(c-t), 
        where c are the raw counts for the O labels, t are the total number of 
        counts per data point, and B is a constant."""

        # If an observation is not voting for anything, we consider it as "O"
        not_voting_obs = (self._get_vote_matrix().sum(axis=1) == 0)
        label_votes[:,0] += label_votes[:,not_voting_obs].sum(axis=1)
        label_votes[:,not_voting_obs] = 0
        
        # Do the normalisation 
        diff = label_votes[:,0] - label_votes.sum(axis=1)
        label_votes[:,0] =  power_base ** diff
        return label_votes



############################################
# Concrete majority voter aggregators
############################################
        
class MajorityVoter(MajorityVoterMixin,TextAggregatorMixin):
    """Aggregator for text classification based on majority voting"""

    def __init__(self, name:str, labels:List[str], 
                 initial_weights:Optional[Dict[str,float]]=None):
        """Creates a new aggregator for text classification using majority
        voting. For each unique span annotated by at least one labelling source, 
        the class constructs a probability distribution over possible labels 
        based on the number of labelling sources "voting" for that label.
        
        Arguments:
        - name is the aggregator name
        - labels is a list of output labels to aggregate. Labels that are not 
          mentioned here are ignored. 
        - initial_weights provides a numeric weight to labelling sources.
          If left unspecified, the class assumes uniform weights.
        """
        AbstractAggregator.__init__(self, name, labels)
        MajorityVoterMixin.__init__(self,initial_weights)
       

class SequentialMajorityVoter(MajorityVoterMixin,SequenceAggregatorMixin):
    """Aggregator for sequence labelling based on majority voting"""
    
    def __init__(self, name:str, labels:List[str], prefixes:str="BIO",
                 initial_weights:Optional[Dict[str,float]]=None):
        """Creates a new aggregator for sequence labelling using majority
        voting. For each token annotated by at least one labelling source, 
        the class constructs a probability distribution over possible labels 
        based on the number of labelling sources "voting" for that label.
        
        Arguments:
        - name is the aggregator name
        - labels is a list of output labels to aggregate. Labels that are not 
        mentioned here are ignored. 
        - prefixes is the tagging scheme to use, such as IO, BIO or BILUO
        - initial_weights provides a numeric weight to labelling sources.
          If left unspecified, the class assumes uniform weights.
        """
        AbstractAggregator.__init__(self, name, labels)
        SequenceAggregatorMixin.__init__(self, prefixes)
        MajorityVoterMixin.__init__(self,initial_weights)     
      
      
       
class MultilabelMajorityVoter(MultilabelAggregatorMixin, MajorityVoterMixin,
                            TextAggregatorMixin,AbstractAggregator):
    
    def __init__(self, name:str, labels:List[str], 
                 initial_weights:Optional[Dict[str,float]]=None):
        """Creates a new, multilabel aggregator for text classification using majority
        voting. For each unique span annotated by at least one labelling source, 
        the class constructs a probability distribution over possible labels 
        based on the number of labelling sources "voting" for that label.
        
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
        MultilabelAggregatorMixin.__init__(self, MajorityVoter, initial_weights=initial_weights)


  
class MultilabelSequentialMajorityVoter(MultilabelAggregatorMixin, SequenceAggregatorMixin,
                                        AbstractAggregator):
    
    def __init__(self, name:str, labels:List[str], prefixes:str="BIO",
                 initial_weights:Optional[Dict[str,float]]=None):
        """Creates a new, multilabel aggregator for sequence labelling 
        using majority voting. For each token annotated by at least one 
        labelling source, the class constructs a probability distribution 
        over possible labels based on the number of labelling sources 
        "voting" for that label.
        
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
        MultilabelAggregatorMixin.__init__(self, SequentialMajorityVoter, initial_weights=initial_weights)
