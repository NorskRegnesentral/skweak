
import json
import re
import functools
from typing import List, Dict, Tuple, Optional, TypeVar, Iterable
from spacy.tokens import Doc, Token, Span, DocBin  # type: ignore
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
    if len(tok) < 2:
        return False

    # If the lemma is titled or in uppercase, just return True
    elif tok.lemma_.istitle() and len(tok.lemma_) >2:
        return True
    elif tok.lemma_.isupper() and len(tok.lemma_) >2 and tok.lemma_ != "-PRON-":
        return True
    # If there is no lemma, but the token is in uppercase, return true as well
    elif tok.lemma_=="" and tok.is_upper:
        return True
    
    # We do not consider the 200 most common words as proper name
    elif (tok.lemma_.islower() and tok.lemma in tok.vocab.strings
          and tok.vocab[tok.lemma].rank < min_rank):
        return False

    # Handling cases such as iPad
    elif len(tok) > 2 and tok.text[0].islower() and tok.text[1].isupper():
        return True

    # Handling cases such as IceFog
    elif (len(tok) > 2 and tok.text[0].isupper()
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
    elif tok.pos_ == "PROPN":
        return True

    # If the token is in lowercase but is a quite rare token
    elif len(tok) > 3 and tok.is_lower and len(tok.vocab.vectors) > 0 and tok.is_oov:
        return True

    return False


def is_infrequent(span: Span, max_rank_threshold=15000) -> bool:
    """Returns true if there is at least one token that is quite infrequent"""

    max_rank = max(tok.rank if len(span.vocab.vectors) >
                   0 and tok.rank > 0 else 0 for tok in span)
    return max_rank > max_rank_threshold


def in_compound(tok: Token):
    """Returns true if the spacy token is part of a compound phrase"""

    if tok.dep_ == "compound":
        return True
    elif tok.i > 0 and tok.nbor(-1).dep_ == "compound":
        return True
    return False


def replace_ner_spans(doc: Doc, source: str):
    """Given a Spacy Doc object and the name of an annotation source, replaces
    the current named entities by the ones specified in the source"""

    # We create Spacy spans based on the annotation layer
    spans = []
    if source in doc.spans:
        for span in doc.spans[source]:
            spans.append(span)
    doc.ents = tuple(spans)

    return doc


@functools.lru_cache(maxsize=5)
def get_spacy_model(spacy_model_name: str):
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

    next_boundary_indices = np.searchsorted(boundaries, range(1, len(doc)+1))
    next_boundaries = [boundaries[i] if i < len(boundaries) else len(doc)
                       for i in next_boundary_indices]
    return next_boundaries


############################################
# I/O related functions
############################################


def docbin_reader(docbin_file_path: str, spacy_model_name: str = "en_core_web_md",
                  cutoff: Optional[int] = None, nb_to_skip: int = 0):
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


def docbin_writer(docs: Iterable[Doc], docbin_output_path: str):
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


def json_writer(docs, json_file_path: str, source: str = None):
    """Converts a collection of Spacy Doc objects to a JSON format,
    such that it can be used to train the Spacy NER model. (for Spacy v2)

    Source must be an aggregated source (defined in user_data["agg_spans"]), which
    will correspond to the target values in the JSON file.
    """
    import spacy
    if int(spacy.__version__[0]) > 2:
        raise RuntimeError("Only supported for Spacy v2")

    import spacy.gold  # type: ignore

    # We start opening up the JSON file
    print("Writing JSON file to", json_file_path)
    out_fd = open(json_file_path, "wt")
    out_fd.write("[{\"id\": 0, \"paragraphs\": [\n")
    for i, doc in enumerate(docs):

        # We replace the NER labels with the annotation source
        if source is not None:
            doc = replace_ner_spans(doc, source)

        # We dump the JSON content to the file
        d = spacy.gold.docs_to_json([doc])
        s = json.dumps(d["paragraphs"]).strip("[]")
        if i > 0:
            s = ",\n" + s
        out_fd.write(s)

        if i > 0 and i % 1000 == 0:
            print("Converted documents:", i)
            out_fd.flush()

    # And finally close all file descriptors
    out_fd.write("]}]\n")
    out_fd.flush()
    out_fd.close()


############################################
# Operations on spans
############################################


def get_spans(doc: Doc, sources: List[str], labels: Optional[List[str]] = None
              ) -> List[Span]:
    """Return the spans annotated by a list of labelling sources. If two
    spans are overlapping, the longest spans are kept. One can also specify the 
    labels to focus on (if empty, we extract all).  """

    # Creating a list of spans
    spans = []
    for source in sources:
        if source in doc.spans:
            for span in doc.spans[source]:
                if labels is None or span.label_ in labels:
                    spans.append(span)
        else:
            raise RuntimeError("Annotation source \"%s\" cannot be found" % source)

    # Remove possible overlaps
    spans = _remove_overlaps(spans)

    return spans


def get_spans_with_probs(doc: Doc, source: str, labels: Optional[List[str]] = None
                         ) -> List[Tuple[Span,float]]:
    """Return the spans annotated by an aggregated source. The method returns a
    dictionary of non-overlapping spans where the keys
    are (start, end) pairs and the values are pairs of (label, prob).
    """

    spans = []
    if source in doc.spans:
        for span in doc.spans[source]:
            if labels is None or span.label_ in labels:
                prob = _get_agg_span_prob(doc, source, span)
                spans.append((span, prob))
    else:
        raise RuntimeError("Annotation source \"%s\" cannot be found" % source)

    return spans


def _get_agg_span_prob(doc, source, span):
    """Get the probability that the source assigns the (start,end)->label span"""

    if source not in doc.spans:
        return 0
    elif "probs" not in doc.spans[source].attrs:
        return 1
    probs = doc.spans[source].attrs["probs"]
    if (span.start, span.end) in probs:
        return probs[(span.start, span.end)]
    probs_per_token = []
    for i in range(span.start, span.end):
        if i in probs:
            for prefixed_label, prob in probs[i].items():
                if prefixed_label.endswith("-%s" % span.label_):
                    probs_per_token.append(prob)
    return sum(probs_per_token)/(len(span))


def count_nb_occurrences(tokens: Tuple[str, ...], all_tokens: List[str]):
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


def at_least_nb_occurrences(tokens: Tuple[str, ...], all_tokens: List[str], min_threshold):
    """Returns true if the number of occurences of the sequence of tokens in the
    full list all_tokens is at least min_threshold, and false otherwise"""

    if len(tokens) == 1:
        return all_tokens.count(tokens[0]) >= min_threshold

    nb_occurrences = 0
    for i in range(len(all_tokens)):
        for k in range(len(tokens)):
            if (i+k) >= len(all_tokens) or all_tokens[i+k] != tokens[k]:
                break
        else:
            nb_occurrences += 1
            if nb_occurrences >= min_threshold:
                return True
    return False


def _remove_overlaps(spans: List[Span]) -> List[Span]:
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
            start1 = spans[i-1].start
            end1 = spans[i-1].end
            start2 = spans[i].start
            end2 = spans[i].end
            if start2 < end1 and start1 < end2:
                length_diff = (end1-start1) - (end2-start2)
                if length_diff > 0:
                    del spans[i]
                else:
                    del spans[i-1]
                finished = False
                break

    return spans


def merge_contiguous_spans(spans: List[Tuple[int, int, str]], doc: Doc,
                           acceptable_gaps: str = ","):
    """Merge spans that are contiguous (and with same label), or only
    separated with some predefined punctuation symbols"""

    finished = False
    while not finished:
        finished = True
        spans.sort()
        for i in range(1, len(spans)):
            start1, end1, label1 = spans[i-1]
            start2, end2, label2 = spans[i]
            if end1 == start2 or (end1 == start2-1 and doc[end1].text in acceptable_gaps):
                if label1 == label2:
                    new_spans = spans[:i-1] if i > 1 else []
                    new_spans.append((start1, end2, label1))
                    new_spans += spans[i+1:]
                    spans = new_spans
                    finished = False
                    break
    return spans


def get_overlaps(start: int, end: int, other_spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Returns a list of overlaps (as (start, end, value) between the provided span
    and other existing spans"""

    overlaps = []
    other_spans.sort()
    start_search, end_search = _binary_search(start, end, other_spans)

    for other_span_start, other_span_end in other_spans[start_search:end_search]:
        if start < other_span_start and end > other_span_end:
            overlaps.append((other_span_start, other_span_end))

    return overlaps


def _binary_search(start: int, end: int, intervals: List[Tuple[int, int]]) -> Tuple[int, int]:
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
                   sources: List[str] = None) -> np.ndarray:
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
            prefix, label = label.split("-", 1)
            prefixes.add(prefix)
            labels_without_prefix.add(label)

    if sources is None:
        sources = list(doc.spans.keys())

    # Creating the numpy array itself
    data = np.zeros((len(doc), len(sources)), dtype=np.int16)

    for source_index, source in enumerate(sources):
        for span in doc.spans.get(source, []):

            if span.label_ not in labels_without_prefix:
                continue

            # If the span is a single token, we can use U
            if "U" in prefixes and len(span) == 1:
                data[span.start, source_index] = label_indices["U-%s" % span.label_]
                continue

            # Otherwise, we use B, I and L
            if "B" in prefixes:
                data[span.start, source_index] = label_indices["B-%s" % span.label_]
            if "I" in prefixes:
                start_i = (span.start+1) if "B" in prefixes else span.start
                end_i = (span.end-1) if "L" in prefixes else span.end
                data[start_i:end_i, source_index] = label_indices["I-%s" % span.label_]
            if "L" in prefixes:
                data[span.end-1, source_index] = label_indices["L-%s" % span.label_]

    return data


def token_array_to_spans(agg_array: np.ndarray,
                         prefix_labels: List[str]) -> Dict[Tuple[int, int], str]:
    """Returns an dictionary of spans corresponding to the aggregated 2D
    array. prefix_labels must be list of prefix labels such as B-PERSON,
    I-ORG etc., of same size as the number of columns in the array."""

    spans = {}
    i = 0
    while i < len(agg_array):

        if np.isscalar(agg_array[i]):
            value_index = agg_array[i]
        else:  # If we have probabilities, select most likely label
            value_index = agg_array[i].argmax()

        if value_index == 0:
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
            spans[(start, i)] = label

    return spans


def token_array_to_probs(agg_array: np.ndarray,
                         prefix_labels: List[str]) -> Dict[int, Dict[str, float]]:
    """Given a 2D array containing, for each token, the probabilities for a
    each possible output label in prefix form (B-PERSON, I-ORG, etc.), returns
    a dictionary of dictionaries mapping token indices to probability distributions
    over their possible labels. The "O" label and labels with zero probabilities
    are ignored.
    """

    # Initialising the label sequence
    token_probs = {}

    # We only look at labels beyond "O", and with non-zero probability
    row_indices, col_indices = np.nonzero(agg_array[:, 1:])
    for i, j in zip(row_indices, col_indices):
        if i not in token_probs:
            token_probs[i] = {prefix_labels[j+1]: agg_array[i, j+1]}
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
                and prefix_label1[2:] == prefix_label2[2:]):
            return True
        elif "U" not in encoding:
            return (prefix_label2 == "O"
                    or prefix_label2.startswith("B-")
                    or prefix_label2.startswith("U-")
                    or (prefix_label2.startswith("I-") and "B" not in encoding))

    elif prefix_label1.startswith("I-"):
        if ((prefix_label2.startswith("I-")
             or prefix_label2.startswith("L-"))
                and prefix_label1[2:] == prefix_label2[2:]):
            return True
        elif "L" not in encoding:
            return (prefix_label2 == "O"
                    or prefix_label2.startswith("B-")
                    or prefix_label2.startswith("U-")
                    or (prefix_label2.startswith("I-") and "B" not in encoding))

    elif prefix_label1 == "O" or prefix_label1.startswith("L-") or prefix_label1.startswith("U-"):
        return (prefix_label2 == "O"
                or prefix_label2.startswith("B-")
                or prefix_label2.startswith("U-")
                or (prefix_label2.startswith("I-") and "B" not in encoding))


############################################
# Visualisation
############################################

def display_entities(doc: Doc, layer=None, add_tooltip=True):
    """Display the entities annotated in a spacy document, based on the
    provided annotation layer(s). If layer is None, the method displays
    the entities from Spacy. 
    This method will only work in a Jupyter Notebook or similar. 
    """
    import spacy.displacy
    import IPython.core.display
    if layer is None:
        spans = doc.ents
    elif type(layer) is list:
        spans = get_spans(doc, layer)
    elif type(layer) == str:
        if "*" in layer:
            matched_layers = [l for l in doc.spans
                              if re.match(layer.replace("*", ".*?")+"$", l)]
            spans = get_spans(doc, matched_layers)
        else:
            spans = doc.spans[layer]
    else:
        raise RuntimeError("Layer type not accepted")

    entities = {}
    for span in spans:

        start_char = doc[span.start].idx
        end_char = doc[span.end-1].idx + len(doc[span.end-1])

        if (start_char, end_char) not in entities:
            entities[(start_char, end_char)] = span.label_

        # If we have several alternative labels for a span, join them with +
        elif span.label_ not in entities[(start_char, end_char)]:
            entities[(start_char, end_char)] = entities[(
                start_char, end_char)] + "+" + span.label_

    entities = [{"start": start, "end": end, "label": label}
                for (start, end), label in entities.items()]
    doc2 = {"text": doc.text, "title": None, "ents": entities}
    html = spacy.displacy.render(doc2, jupyter=False, style="ent", manual=True)

    if add_tooltip and type(layer)==str and "sources" in doc.spans[layer].attrs:
        html = _enrich_with_tooltip(doc, html, doc.spans[layer].attrs["sources"])  # type: ignore

    ipython_html = IPython.core.display.HTML(
        '<span class="tex2jax_ignore">{}</span>'.format(html))
    return IPython.core.display.display(ipython_html)


def _enrich_with_tooltip(doc: Doc, html: str, sources: List[str]):
    """Enrich the HTML produced by spacy with tooltips displaying the predictions
    of each labelling function"""

    import spacy.util
    if len(doc.spans)==0:
        return html

    # Retrieves annotations for each token
    annotations_by_tok = {}
    for source in sources:
        for span in doc.spans[source]:
            for i in range(span.start, span.end):
                annotations_by_tok[i] = annotations_by_tok.get(i, []) + [(source, span.label_)]

    # We determine which characters are part of the HTML markup and not the text
    all_chars_to_skip = set()
    for fragment in re.finditer("<span.+?</span>", html):
        all_chars_to_skip.update(range(fragment.start(0), fragment.end(0)))
    for fragment in re.finditer("</?div.*?>", html):
        all_chars_to_skip.update(range(fragment.start(0), fragment.end(0)))
    for fragment in re.finditer("</?mark.*?>", html):
        all_chars_to_skip.update(range(fragment.start(0), fragment.end(0)))

    # We loop on each token
    curr_pos = 0
    new_fragments = []
    for tok in doc:

        # We search for the token position in the HTML
        toktext = spacy.util.escape_html(tok.text)
        if "\n" in toktext:
            continue
        start_pos = html.index(toktext, curr_pos)
        if start_pos == -1:
            raise RuntimeError("could not find", tok)
        while any((i in all_chars_to_skip for i in range(start_pos, start_pos + len(toktext)))):
            start_pos = html.index(toktext, start_pos+1)
            if start_pos == -1:
                raise RuntimeError("could not find", tok)

        # We add the preceding fragment
        new_fragments.append(html[curr_pos:start_pos])

        # If the token has annotations, we create a tooltip
        if tok.i in annotations_by_tok:
            lines = ["%s:\t%s&nbsp;&nbsp" %
                     (ann, label) for ann, label in annotations_by_tok[tok.i]]
            max_width = 7*max([len(l) for l in lines])
            new_fragment = ("<label class='tooltip'>%s" % toktext +
                            "<span class='tooltip-text' style='width:%ipx'>"%max_width +
                            "%s</span></label>" %"<br>".join(lines))
        else:
            new_fragment = toktext
        new_fragments.append(new_fragment)
        curr_pos = start_pos + len(toktext)

    new_fragments.append(html[curr_pos:])

    new_html = """<style>
.tooltip {  position: relative;  border-bottom: 1px dotted black; }
.tooltip .tooltip-text {visibility: hidden;  background-color: black;  color: white;
                        line-height: 1.2;  text-align: right;  border-radius: 6px;
                        padding: 5px 0; position: absolute; z-index: 1; margin-left:1em;
                        opacity: 0; transition: opacity 1s;}
.tooltip .tooltip-text::after {position: absolute; top: 1.5em; right: 100%; margin-top: -5px;
                               border-width: 5px; border-style: solid; 
                               border-color: transparent black transparent transparent;}
.tooltip:hover .tooltip-text {visibility: visible; opacity: 1;}
</style>
""" + "".join(new_fragments)

    return new_html
