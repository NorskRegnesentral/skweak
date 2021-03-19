from __future__ import annotations
import json, re, functools
from typing import List, Set, Dict, Tuple, Optional, TypeVar
from spacy.tokens import Doc, Token, Span, DocBin
import numpy as np

T = TypeVar('T')
    
############################################
# Utility functions for NLP analysis
############################################
  
def is_likely_proper(tok: Token, min_rank=200) -> bool:
    """Returns true if the spacy token is a likely proper name, based on its form.
    
    NB: this method should only be used for languages that have a distinction between
    lowercase and uppercase (so called bicameral scripts)."""
      
    # We require at least two characters
    if len(tok)< 2:
        return False
    
    # If the lemma is titled or in uppercase, just return True
    elif tok.lemma_.istitle() or (tok.lemma_.isupper() and tok.lemma_!="-PRON-"):
        return True
   
    # We do not consider the 200 most common words as proper name
    elif (tok.lemma_.islower() and tok.lemma in tok.vocab.strings 
          and tok.vocab[tok.lemma].rank < min_rank):
        return False
    
    # Handling cases such as iPad
    elif len(tok)>2 and tok.text[0].islower() and tok.text[1].isupper():
        return True
    
    # Handling cases such as IceFog
    elif (len(tok)>2 and tok.text[0].isupper() 
          and any([k.islower() for k in tok.text[1:]])
          and any([k.isupper() for k in tok.text[1:]])):
        return True
       
    # Else, check whether the surface token is titled and is not sentence-initial
    # NB: This should be commented out for languages such as German
    elif (tok.i > 0 and tok.is_title and not tok.is_sent_start 
          and tok.nbor(-1).text not in {'\'', '"', '‘', '“', '”', '’', "\n", "|"} 
          and not tok.nbor(-1).text.endswith(".")):
        return True
    
    # If the part-of-speech is a proper noun
    elif tok.pos_=="PROPN":
        return True
    
    # If the token is in lowercase but is a quite rare token
    elif len(tok)>3 and tok.is_lower and len(tok.vocab.vectors) > 0 and tok.is_oov:
        return True
    
    return False  
                           

def is_infrequent(span: Span, max_rank_threshold=15000) -> bool:  
    """Returns true if there is at least one token that is quite infrequent"""
       
    max_rank = max(tok.rank if len(span.vocab.vectors) > 0 and tok.rank > 0 else 0 for tok in span)
    return max_rank > max_rank_threshold


def in_compound(tok: Token):
    """Returns true if the spacy token is part of a compound phrase"""
 
    if tok.dep_=="compound":
        return True
    elif tok.i > 0 and tok.nbor(-1).dep_=="compound":
        return True
    return False


def replace_ner_spans(doc: Doc, source: str):
    """Given a Spacy Doc object and the name of an annotation source, replaces
    the current named entities by the ones specified in the source"""
    
    
    # We create Spacy spans based on the annotation layer
    spans = []
    if source in doc.user_data["spans"]:
        for (start, end), label in get_spans(doc, [source]).items():
            spans.append(Span(doc, start, end, label))
    elif source in doc.user_data["agg_spans"]:
        for (start, end), (label, prob) in get_agg_spans(doc, source).items():
            spans.append(Span(doc, start, end, label))
        
    doc.ents = tuple(spans)
    
    return doc


@functools.lru_cache(maxsize=5)
def get_spacy_model(spacy_model_name:str):
    """Returns the vocabulary associated with the spacy model 
    (and caches it for faster access)"""
    
    import spacy
    return spacy.load(spacy_model_name)


@functools.lru_cache(maxsize=1)
def get_tokens(doc: Doc) -> List[str]:
    """Returns the list of tokens from a given spacy Document. As it is an
    operation that (for some strange reason) actually takes some CPU resources,
    we cache the results, as it is a frequent operation, e.g. for gazetteers. """
    
    return [tok.text for tok in doc]


@functools.lru_cache(maxsize=1)
def get_next_sentence_boundaries(doc: Doc) -> List[int]:
    """Returns a list of integers (of same size as the number of tokens) 
    expressing, for each token, the position of the next sentence boundary
    (start-of-sentence token). """
    
    boundaries = []
    for tok in doc:
        if tok.is_sent_start:
            boundaries.append(tok.i)
            
    next_boundary_indices = np.searchsorted(boundaries, range(1,len(doc)+1))
    next_boundaries = [boundaries[i] if i < len(boundaries) else len(doc) 
                       for i in next_boundary_indices]
    return next_boundaries


############################################
# I/O related functions
############################################

   
def docbin_reader(docbin_file_path: str, spacy_model_name:str = "en_core_web_md", 
                  cutoff:Optional[int]=None, nb_to_skip:int=0):
    """Read a binary file containing a DocBin repository of spacy documents.
    In addition to the file path, we also need to provide the name of the spacy
    model (which is necessary to load the vocabulary), such as "en_core_web_md".
    
    If cutoff is specified, the method will stop after generating the given
    number of documents. If nb_to_skip is > 0, the method will skip the given
    number of documents before starting the generation.
    """
 
    import spacy
        
    # Reading the binary data from the file       
    fd = open(docbin_file_path, "rb")
    data = fd.read()
    fd.close()
    docbin = DocBin(store_user_data=True)
    docbin.from_bytes(data)
    del data
#    print("Total number of documents in docbin:", len(docbin))

    # Skip a number of documents
    if nb_to_skip:
        docbin.tokens = docbin.tokens[nb_to_skip:]
        docbin.spaces = docbin.spaces[nb_to_skip:]
        docbin.user_data = docbin.user_data[nb_to_skip:]

    # Retrieves the vocabulary
    vocab = get_spacy_model(spacy_model_name).vocab
    
    # We finally generate the documents one by one
    reader = docbin.get_docs(vocab) 
    for i, doc in enumerate(reader):
        yield doc
        if cutoff is not None and (i+1) >= cutoff:
            return


def docbin_writer(docs: List, docbin_output_path: str):
    """Writes a stream of Spacy Doc objects to a binary file in the DocBin format."""
    
    import spacy.attrs
    # Creating the DocBin object (with all attributes)
    attrs = [spacy.attrs.LEMMA, spacy.attrs.TAG, spacy.attrs.DEP, spacy.attrs.HEAD, 
                 spacy.attrs.ENT_IOB, spacy.attrs.ENT_TYPE]
    docbin = DocBin(attrs=attrs, store_user_data=True)

    # Storing the documents in the DocBin repository
    for doc in docs:
        doc.cats = {}
        docbin.add(doc)
    data = docbin.to_bytes()   
    
    # And writing the content to the file
    print("Write to", docbin_output_path, end="...", flush=True)
    fd = open(docbin_output_path, "wb")
    fd.write(data)
    fd.close()
    print("done")

                   
def json_writer(docs, json_file_path: str, source: str="hmm"):
    """Converts a collection of Spacy Doc objects to a JSON format,
    such that it can be used to train the Spacy NER model. 
    
    Source must be an aggregated source (defined in user_data["agg_spans"]), which 
    will correspond to the target values in the JSON file. 
    """
    
    import spacy.gold
    
    #We start opening up the JSON file
    print("Writing JSON file to", json_file_path)
    out_fd = open(json_file_path, "wt")
    out_fd.write("[{\"id\": 0, \"paragraphs\": [\n")
    for i, doc in enumerate(docs):
        
        # We replace the NER labels with the annotation source
        doc = replace_ner_spans(doc, source)
        
        # We dump the JSON content to the file
        d = spacy.gold.docs_to_json([doc])
        s = json.dumps(d["paragraphs"]).strip("[]")
        if i > 0:
            s = ",\n" + s
        out_fd.write(s)

        if i>0 and i % 1000 == 0:
            print("Converted documents:", i)
            out_fd.flush() 
    
    # And finally close all file descriptors
    out_fd.write("]}]\n")
    out_fd.flush()
    out_fd.close()
    
    
############################################
# Operations on spans
############################################
      
                
def get_spans(doc: Doc, sources: List[str], labels: List[str]=None):
    """Return the spans annotated by a list of labelling sources. If two 
    spans are overlapping, the longest spans are kept.
    
    One can also specify the labels to focus on (if empty, we extract
    all). The method returns a dictionary of non-overlapping spans where the keys
    are (start, end) pairs and the values are the corresponding labels. 
    """
    
    # Creating a list of spans
    spans = []
    for source in sources:
        if source in doc.user_data.get("spans", []):
            for (start, end), label in doc.user_data["spans"][source].items():
                if not labels or label in labels:
                    spans.append((start, end, label))
        elif source in doc.user_data.get("agg_spans", []):
            for (start, end), (label, _) in get_agg_spans(doc, source, labels).items():
                spans.append((start,end,label))
        else:
            raise RuntimeError("Annotation source \"%s\" cannot be found"%source)
                            
    spans = remove_overlaps(spans)
    
    spans = {(start,end):label for start, end, label in spans}
    return spans


def get_agg_spans(doc: Doc, agg_source: str, labels: List[str]=None):
    """Return the spans annotated by an aggregated source. The method returns a 
    dictionary of non-overlapping spans where the keys
    are (start, end) pairs and the values are pairs of (label, prob). 
    """
       
    spans = {}
    if agg_source in doc.user_data.get("agg_spans", []):
        for (start, end), label in doc.user_data["agg_spans"][agg_source].items():
            if not labels or label in labels:
                    
                prob = get_agg_span_prob(doc, agg_source, start, end, label)
                spans[(start, end)] = (label, prob)
    elif agg_source in doc.user_data["spans"]:
        for (start,end), label in get_spans(doc, [agg_source], labels).items():
            spans[(start, end)] = (label, 1.0)
    else:
        raise RuntimeError("Annotation source \"%s\" cannot be found"%agg_source)
    
    return spans

    


def get_agg_span_prob(doc, source, start, end, label):
    """Get the probability that the source assigns the (start,end)->label span"""
    
    if source not in doc.user_data["agg_probs"]:
        return 0
    agg_probs = doc.user_data["agg_probs"][source]
    if (start, end) in agg_probs:
        return agg_probs[(start, end)].get(label, 0.0)
    probs_per_token = []
    for i in range(start, end):
        if i in agg_probs:
            for prefixed_label, prob in agg_probs[i].items():
                if prefixed_label.endswith("-%s"%label):
                    probs_per_token.append(prob)
    return sum(probs_per_token)/(end-start)
                

def count_nb_occurrences(tokens: Tuple[str,...], all_tokens: List[str]):
    """Count the number of occurences of the sequence of tokens in the
    full list all_tokens"""
    
    nb_occurrences = 0
    for i in range(len(all_tokens)):
        for k in range(len(tokens)):
            if all_tokens[i+k] != tokens[k]:
                break
        else:
            nb_occurrences += 1
    return nb_occurrences        

def at_least_nb_occurrences(tokens: Tuple[str,...], all_tokens: List[str], min_threshold):
    """Returns true if the number of occurences of the sequence of tokens in the
    full list all_tokens is at least min_threshold, and false otherwise"""
    
    if len(tokens)==1:
        return all_tokens.count(tokens[0]) >= min_threshold
    
    nb_occurrences = 0
    for i in range(len(all_tokens)):
        for k in range(len(tokens)):
            if all_tokens[i+k] != tokens[k]:
                break
        else:
            nb_occurrences += 1
            if nb_occurrences >= min_threshold:
                return True
    return False      


def remove_overlaps(spans: List[Tuple[int, int, str]]
                    ) -> List[Tuple[int, int, str]]:
    """Remove overlaps between spans expressed as (start, end, label, score)
    tuples. When two overlapping spans are detected, the method keeps the
    longest span and removes the other. If the two scores are identical, 
    the first span is discarded).
    """
    
    # We sort the spans by their position        
    spans.sort() 
    
    # We resolve overlaps between spans
    finished = False  
    while not finished:
        finished = True
        for i in range(1, len(spans)):
            
            # If two spans are overlapping , keep the longest one
            start1, end1, _ = spans[i-1]
            start2, end2, _ = spans[i]
            if start2 < end1 and start1 < end2:
                length_diff = (end1-start1) - (end2-start2)
                if length_diff > 0:
                    del spans[i]
                else:
                    del spans[i-1]
                finished = False
                break
            
    return spans
    
 
def merge_contiguous_spans(spans: List[Tuple[int,int,str]], doc: Doc, 
                           acceptable_gaps:str = ","):
    """Merge spans that are contiguous (and with same label), or only 
    separated with some predefined punctuation symbols"""
    
    finished = False   
    while not finished:
        finished = True
        spans.sort()
        for i in range(1, len(spans)):
            start1, end1, label1 = spans[i-1]
            start2, end2, label2 = spans[i]
            if end1==start2 or (end1==start2-1 and doc[end1].text in acceptable_gaps):
                if label1==label2:
                    new_spans = spans[:i-1] if i>1 else []
                    new_spans.append((start1, end2, label1))
                    new_spans += spans[i+1:]
                    spans = new_spans
                    finished = False
                    break
    return spans


def get_overlaps(start:int, end:int, other_spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Returns a list of overlaps (as (start, end, value) between the provided span 
    and other existing spans"""
    
    overlaps = []
    other_spans.sort()
    start_search, end_search = _binary_search(start, end, other_spans)
        
    for other_span_start, other_span_end in other_spans[start_search:end_search]:
        if start < other_span_start and end > other_span_end:
            overlaps.append((other_span_start, other_span_end))

    return overlaps
   

def _binary_search(start:int, end:int, intervals: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Performs a binary search"""
    
    start_search = 0
    end_search = len(intervals)
    while start_search < (end_search-1):        
        mid = start_search + (end_search-start_search)//2
        (interval_start, interval_end) = intervals[mid]
        
        if interval_end <= start:
            start_search = mid
        elif interval_start >= end:
            end_search = mid
        else:
            break
    return start_search, end_search


def get_subsequences(sequence: List[T]) -> List[List[T]]:
    """Returns the list of possible subsequences that are included
    in the full sequence (including the original sequence)."""
    
    subsequences = []
    for length in range(1, len(sequence)+1):
        for i in range(length, len(sequence)+1):
            subsequences.append(sequence[i-length:i])
    return subsequences


def spans_to_array(doc: Doc, labels: List[str], 
                    sources: Set[str]=None) -> np.ndarray:
    """Convert the annotations of a spacy document into a 2D array.
    Each row corresponds to a token, and each column to a labelling
    source. In other words, the value at (i,j) represents the prediction
    of source j for token i. This prediction is expressed as the
    index of the label in the labels.
    
    Labels must be a list of labels (such as B-PERSON, I-ORG) to detect. 
    Sources should be a list of labelling sources. If empty, all sources
    are employed. 
    
    NB: we assume the labels use either IO/BIO/BILUO, and that the
    O label is at position 0. 
    """

    # Creating some helper dictionaries
    label_indices = {}
    prefixes = set()
    labels_without_prefix = set()
    for i, label in enumerate(labels):
        label_indices[label] = i
        if "-" in label:
            prefix, label = label.split("-",1)
            prefixes.add(prefix)
            labels_without_prefix.add(label)
    
    if sources is None:
        sources = list(doc.user_data.get("spans", {}).keys())

    # Creating the numpy array itself
    data  = np.zeros((len(doc), len(sources)), dtype=np.int16)

    for source_index, source in enumerate(sources):
        for (start,end),label in doc.user_data["spans"].get(source, {}).items():
            
            if label not in labels_without_prefix:
                continue
            
            # If the span is a single token, we can use U
            if "U" in prefixes and (end-start)==1:
                data[start, source_index] = label_indices["U-%s"%label]
                continue
                    
            # Otherwise, we use B, I and L
            if "B" in prefixes:
                data[start, source_index] = label_indices["B-%s"%label]
            if "I" in prefixes:
                start_i = (start+1) if "B" in prefixes else start
                end_i = (end-1) if "L" in prefixes else end
                data[start_i:end_i, source_index] = label_indices["I-%s"%label]
            if "L" in prefixes:
                data[end-1, source_index] = label_indices["L-%s"%label]

    return data

        
def token_array_to_spans(agg_array: np.ndarray, 
                   prefix_labels: List[str]) -> Dict[Tuple[int,int], str]:
    """Returns an dictionary of spans corresponding to the aggregated 2D
    array. prefix_labels must be list of prefix labels such as B-PERSON,
    I-ORG etc., of same size as the number of columns in the array."""
    
    spans = {}    
    i = 0
    while i < len(agg_array):
          
        if np.isscalar(agg_array[i]):
            value_index = agg_array[i]
        else: # If we have probabilities, select most likely label
            value_index = agg_array[i].argmax()
        
        if value_index ==0:
            i += 1
            continue
            
        prefix_label = prefix_labels[value_index]
        prefix, label = prefix_label.split("-", 1)
        
        # If the prefix is "U", create a single-token span
        if prefix == "U":
            spans[(i, i+1)] = label
            i += 1
            
        # Otherwise, we need to continue until the span ends
        elif prefix in {"B", "I"}:
            start = i
            i += 1
            while i < len(agg_array):
                if np.isscalar(agg_array[i]):
                    next_val = agg_array[i]
                else:
                    next_val = agg_array[i].argmax()
                if next_val == 0:
                    break
                next_prefix_label = prefix_labels[next_val]
                next_prefix, next_label = next_prefix_label.split("-", 1)
                if next_prefix not in {"I", "L"}:
                    break
                i += 1
            spans[(start,i)] = label
    
    return spans


def token_array_to_probs(agg_array: np.ndarray, 
                            prefix_labels: List[str]) -> Dict[int,Dict[str,float]]:
    """Given a 2D array containing, for each token, the probabilities for a 
    each possible output label in prefix form (B-PERSON, I-ORG, etc.), returns
    a dictionary of dictionaries mapping token indices to probability distributions
    over their possible labels. The "O" label and labels with zero probabilities
    are ignored.
    """
    
    # Initialising the label sequence
    token_probs = {}
    
    # We only look at labels beyond "O", and with non-zero probability
    row_indices, col_indices = np.nonzero(agg_array[:,1:])
    for i, j in zip(row_indices, col_indices):
        if i not in token_probs:
            token_probs[i] = {prefix_labels[j+1]:agg_array[i, j+1]}
        else:
            token_probs[i][prefix_labels[j+1]] = agg_array[i, j+1]
        
    return token_probs

                    
def is_valid_start(prefix_label, encoding="BIO"):
    """Returns whether the prefix label is allowed to start a sequence"""
    
    return (prefix_label == "O" 
            or prefix_label.startswith("B-") 
            or prefix_label.startswith("U-") or 
            (prefix_label.startswith("I-") and "B" not in encoding))


def is_valid_transition(prefix_label1, prefix_label2, encoding="BIO"):
    """Returns whether the two labels (associated with a prefix, such as B-PERSON, 
    I-ORG etc.) are allowed to follow one another according to the encoding (which 
    can be BIO, BILUO, IO, etc.)"""
    
    if prefix_label1.startswith("B-"):
        if ((prefix_label2.startswith("I-")
             or prefix_label2.startswith("L-"))
            and prefix_label1[2:]==prefix_label2[2:]):
            return True
        elif "U" not in encoding:
            return (prefix_label2 == "O" 
                or prefix_label2.startswith("B-") 
                or prefix_label2.startswith("U-") 
                or (prefix_label2.startswith("I-") and "B" not in encoding))
                    
    elif prefix_label1.startswith("I-"):
        if ((prefix_label2.startswith("I-") 
             or prefix_label2.startswith("L-"))
            and prefix_label1[2:]==prefix_label2[2:]):
            return True
        elif "L" not in encoding:
            return (prefix_label2 == "O" 
                or prefix_label2.startswith("B-") 
                or prefix_label2.startswith("U-") 
                or (prefix_label2.startswith("I-") and "B" not in encoding))
                 
    elif prefix_label1=="O" or prefix_label1.startswith("L-") or prefix_label1.startswith("U-"):
        return (prefix_label2 == "O" 
                or prefix_label2.startswith("B-") 
                or prefix_label2.startswith("U-") 
                or (prefix_label2.startswith("I-") and "B" not in encoding))
       
        
   
############################################
# Visualisation
############################################


def display_entities(doc: Doc, layer=None):
    """Display the entities annotated in a spacy document, based on the 
    provided annotation layer(s). If layer is None, the method displays
    the entities from Spacy.
    """
    
    import spacy.displacy
    if layer is None:
        spans = {(ent.start,ent.end):ent.label_ for ent in doc.ents}
    elif type(layer) is list:
        spans = get_spans(doc, layer)
    elif type(layer)==str:
        if "*" in layer:
            matched_layers = [l for l in doc.user_data["spans"] 
                              if re.match(layer.replace("*", ".*?")+"$", l)]
            spans = get_spans(doc, matched_layers)
        else:
            spans = get_spans(doc, [layer])
    else:
        raise RuntimeError("Layer type not accepted")
    
    text = doc.text
   
    entities = {}
    for (start,end), label in sorted(spans.items()):
        
        start_char = doc[start].idx 
        end_char = doc[end-1].idx + len(doc[end-1])
            
        if (start_char,end_char) not in entities:
            entities[(start_char, end_char)] = label
            
        # If we have several alternative labels for a span, join them with +
        elif label not in entities[(start_char,end_char)]:
            entities[(start_char,end_char)] = entities[(start_char,end_char)]+ "+" + label

    entities = [{"start":start, "end":end, "label":label} for (start,end), label in entities.items()]
    doc2 = {"text":text, "title":None, "ents":entities}
    spacy.displacy.render(doc2, jupyter=True, style="ent", manual=True)


