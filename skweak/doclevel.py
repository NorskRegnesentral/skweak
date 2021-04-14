from __future__ import annotations

from skweak.gazetteers import GazetteerAnnotator
from skweak import utils
from typing import Dict, List, Tuple
from . import gazetteers, base
from spacy.tokens import Doc #type: ignore
from collections import defaultdict

class DocumentHistoryAnnotator(base.BaseAnnotator):
    """Annotation based on the document history: 
    1) if a person name has been mentioned in full (at least two consecutive tokens, 
    most often first name followed by last name), then mark future occurrences of the 
    last token (last name) as a PER as well. 
    2) if an organisation has been mentioned together with a legal type, mark all other 
    occurrences (possibly without the legal type at the end) also as a COMPANY.
    """  
      
    def __init__(self, basename:str, other_name: str, labels: List[str]):
        """Creates a new annotator looking at the global document context, based on another 
        annotation layer (typically a layer aggregating existing annotations). Only the 
        labels specified in the argument will be taken into account."""
        
        super(DocumentHistoryAnnotator, self).__init__(basename)
        self.other_name = other_name
        self.labels = labels
        
       
    def __call__(self, doc: Doc) -> Doc:
        """Search for spans on one single document"""
        
        # Clears existing annotations
        if "spans" not in doc.user_data:
            doc.user_data["spans"] = {}
        for source in doc.user_data["spans"]:
            if source.startswith(self.name):
                doc.user_data["spans"][source] = {}

        # Extract the first mentions of each entity
        first_observed = self.get_first_mentions(doc)

        # We construct tries based on the first mentions
        tries = {label: gazetteers.Trie() for label in self.labels}
        for start, end, label in first_observed:
            tries[label].add([tok.text for tok in doc[start:end]])

        for label, trie in tries.items():

            # For each trie, we create two gazeetters searching for variants of these first
            # mentions (respectively for case-sensitive and case-insensitive search)
            base_name = "%s_%s"%(self.name, label.lower())
            gazetteer1 = GazetteerAnnotator("%s_cased"%base_name, trie, label)
            doc.user_data["spans"][gazetteer1.name] = {}
            for start, end, label in gazetteer1.find_spans(doc):
                if (start, end, label) not in first_observed :
                    doc.user_data["spans"][gazetteer1.name][(start,end)] = label

            gazetteer2 = GazetteerAnnotator("%s_uncased"%base_name, trie, label, 
                                            case_sensitive=False)    
            doc.user_data["spans"][gazetteer2.name] = {}
            for start, end, label in gazetteer2.find_spans(doc):
                if (start, end, label) not in first_observed :
                    doc.user_data["spans"][gazetteer2.name][(start,end)] = label
        return doc 


    def get_first_mentions(self, doc):
        """Returns a set containing the first mentions of each entity as triples
        (start, end, label) according to the "other_name' layer.

        The first mentions also contains subsequences: for instance, a named entity
        "Pierre Lison" will also contain the first mentions of ['Pierre'] and ['Lison'].
        """
        first_observed = {}
        spans = utils.get_agg_spans(doc, self.other_name, self.labels)
        for (start, end), (label, _) in sorted(spans.items()):
            
            # We only consider entities with at least two tokens
            if end-start >= 2:
                
                # We also extract subsequences
                for length in range(1, end-start+1):
                    for i in range(length, end-start+1):

                        start2 = start + i-length
                        end2 = start + i
                        subsequence = tuple(tok.text for tok in doc[start2:end2])

                        # We only consider first mentions
                        if subsequence in first_observed:
                            continue

                        # To avoid FPs, at leat one token must look like a proper name, 
                        # and the mention must have at least 4 characters
                        elif (any(utils.is_likely_proper(tok) for tok in doc[start2:end2]) 
                              and sum(len(tok) for tok in subsequence) > 3):
                            first_observed[subsequence] = (start2, end2, label)

        return set(first_observed.values())

                      

class DocumentMajorityAnnotator(base.BaseAnnotator):
    """Annotation based on majority label for the same entity string elsewhere in the 
    document. The annotation creates two layers for each label, one for case-sensitive 
    occurrences of the entity string in the document, and one for case-insensitive 
    occurrences.
    """
    
    def __init__(self, basename:str, other_name:str):
        """Creates a new annotator that looks at (often aggregated) annotations from
        another layer, and annotates entities based on their majority label elsewhere
        in the document. """
        
        super(DocumentMajorityAnnotator, self).__init__(basename)
        self.other_name = other_name

    
    def __call__(self, doc: Doc) -> Doc:
        """Generates span annotations for one single document based on 
        majority labels"""     

        # Clears existing annotations
        for source in list(doc.user_data["spans"]):
            if source.startswith(self.name+"_"):
                del doc.user_data["spans"][source]

        # We search for the majority label for each entity string
        majority_labels = self.get_majority_labels(doc)

        # we build trie to easily search for these entities in the text
        tries = {label:gazetteers.Trie() for label in set(majority_labels.values())}
        for ent_tokens, label in majority_labels.items():
            tries[label].add(list(ent_tokens))        
        
        # We run two gazeteers (case-sensitive or not) for each label
        for label, trie in tries.items():
            base_name = "%s_%s"%(self.name, label.lower())
            gazetteer1 = GazetteerAnnotator("%s_cased"%base_name, trie, label)
            gazetteer2 = GazetteerAnnotator("%s_uncased"%base_name, trie, label,
                                            case_sensitive=False)
            doc = gazetteer1(doc)
            doc = gazetteer2(doc)
                  
        return doc

    
    def get_majority_labels(self, doc: Doc) -> Dict[Tuple[str], str] :
        """Given a document, searches for the majority label for each entity string
        with at least self.min_counts number of occurrences. """

        # Get the counts for each label per entity string
        # (and also for each form, to take various casings into account)
        label_counts = defaultdict(dict)
        form_counts = defaultdict(dict)
        spans = utils.get_agg_spans(doc, self.other_name)
        
        all_tokens_low = [tok.lower_ for tok in doc]
        checked = {}
        for (start, end), (label, prob) in spans.items():
            
            # We only apply document majority for strings occurring more than once
            tokens_low = tuple(all_tokens_low[start:end])  
            if tokens_low not in checked:
                occurs_several_times = utils.at_least_nb_occurrences(tokens_low, all_tokens_low, 2)
                checked[tokens_low] = occurs_several_times
            else:
                occurs_several_times = checked[tokens_low]
            
            # If the string occurs more than once, update the counts
            if occurs_several_times:
                
                label_counts[tokens_low][label] = label_counts[tokens_low].get(label, 0) + prob
                tokens = tuple(tok.text for tok in doc[start:end])           
                form_counts[tokens_low][tokens] = form_counts[tokens_low].get(tokens, 0) + prob
        
        # Search for the most common label for each entity string
        majority_labels = {}
        for lower_tokens, labels_for_ent in label_counts.items():
            majority_label = max(labels_for_ent, key=lambda x: labels_for_ent[x])
            forms_for_ent = form_counts[lower_tokens]                
            majority_form = max(forms_for_ent, key=lambda x: forms_for_ent[x])
            
            majority_labels[majority_form] = majority_label

        return majority_labels
                