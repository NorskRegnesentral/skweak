import json
import re
from typing import Iterable, List, Dict, Tuple, Optional
from . import base, utils
from spacy.tokens import Doc, Span, Token  # type: ignore
import gzip

############################################
# Gazetteer annotator
############################################



class GazetteerAnnotator(base.SpanAnnotator):
    """Annotation using a gazetteer, i.e. a large list of entity terms. The annotation can
    look at either case-sensitive and case-insensitive occurrences.  The annotator relies 
    on a token-level trie for efficient search. """

    def __init__(self, name: str, tries: Dict[str, 'Trie'], case_sensitive: bool = True, 
                 lookahead: int = 10, additional_checks: bool=True):
        """Creates a new gazeteer, based on:
        - a trie
        - an output label associated with the trie
        - a flag indicating whether the gazetteer should be case-sensitive or not
        - the maximum size of the lookahead window
        - a flag indicating whether to do additional checks to reduce the 
          number of false positives when searching for named entities"""

        super(GazetteerAnnotator, self).__init__(name)

        self.tries = tries
        self.case_sensitive = case_sensitive
        self.lookahead = lookahead
        self.additional_checks = additional_checks
        

    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:
        """Search for occurrences of entity terms in the spacy document"""

        # We extract the tokens (as list of strings)
        tokens = utils.get_tokens(doc)

        # We extract the (token-level) indices for next sentence boundaries
        next_sentence_boundaries = utils.get_next_sentence_boundaries(doc)

        i = 0
        while i < len(doc):

            tok = doc[i]

            # We create a lookahead window starting at the token
            lookahead_length = self._get_lookahead(tok, next_sentence_boundaries[i])

            if lookahead_length:

                window = tokens[i:i+lookahead_length]
                
                matches = []
                # We loop on all tries (one per label)
                for label, trie in self.tries.items():
                    
                    # We search for the longest match
                    match = trie.find_longest_match(window, self.case_sensitive)
                    if match:
                        
                        # We check whether the match is valid
                        if (not self.additional_checks or 
                            self._is_valid_match(doc[i:i+len(match)], match)):
                            matches.append((match, label))
                
                # We choose the longest match(es)
                if matches:
                    max_length = max(len(match) for match, _ in matches)
                    for match, label in matches:
                        if len(match)==max_length:
                            yield i, i+max_length, label

                    # We skip the text until the end of the match
                    i += max_length-1
                    
            i += 1

    def _get_lookahead(self, token: Token, next_sentence_boundary: int) -> int:
        """Returns the longest possible span starting with the current token, and
        satisfying the three following criteria:
        - the maximum length of the span is self.lookahead
        - the span cannot start with a punctuation symbol or within a compound phrase
        - the span cannot cross sentence boundaries
        """

        if token.is_punct:
            return 0
        elif token.i > 0 and token.nbor(-1).dep_ == "compound" and token.nbor(-1).head == token:
            return 0

        return min(next_sentence_boundary-token.i, self.lookahead)

    def _is_valid_match(self, match_span: Span, ent_tokens: List[str]) -> bool:
        """Checks whether the match satisfies the following criteria:
        - the match does not end with a punctuation symbol or within a compound phrase
          (with a head that looks like a proper name)
        - if the actual tokens of the entity contains tokens in "title" case, the match
          must contain at least one token that looks like a proper name
          (to avoid too many false positives).
        """

        last_token = match_span[-1]
        if last_token.is_punct:
            return False
        elif match_span.end < len(match_span.doc):
            if (last_token.dep_ == "compound" and last_token.head.i > last_token.i
                    and utils.is_likely_proper(last_token.head)):
                return False

        if (any(tok.istitle() for tok in ent_tokens) and
                not any(utils.is_likely_proper(tok) for tok in match_span)):
            return False
        return True


############################################
# Trie data structure (used for gazetteers)
############################################

class Trie:
    """Implementation of a trie for searching for occurrences of terms in a text. 

    Internally, the trie is made of nodes expressed as (dict, bool) pairs, where the
    dictionary expressed possible edges (tokens) going out from the node, and the boolean
    indicates whether the node is terminal or not. 
    """

    def __init__(self, entries: List[List[str]] = None):
        """Creates a new trie. If provided, entries must be a list of tokenised entries"""

        self.start = {}
        self.len = 0

        if entries is not None:
            for entry in entries:
                self.add(entry)

    def find_longest_match(self, tokens: List[str], case_sensitive=True) -> List[str]:
        """Search for the longest match (that is, the longest element in the trie that matches
        a prefix of the provided tokens). The tokens must be expressed as a list of strings. 
        The method returns the match as a list of tokens, which is empty is no match could
        be found. 

        If case_sensitive is set to False, the method also checks for matches of alternative 
        casing of the words (lowercase, uppercase and titled)
        """

        edges = self.start
        prefix_length = 0
        matches = []

        for i, token in enumerate(tokens):

            match = self._find_match(token, edges, case_sensitive)
            if match:
                edges, is_terminal = edges[match]
                matches.append(match)
                if is_terminal:
                    prefix_length = i+1
            else:
                break

        return matches[:prefix_length]

    def _find_match(self, token: str, branch: Dict, case_sensitive: bool) -> Optional[str]:
        """Checks whether the token matches any edge in the branch. If yes, 
        returns the match (which can be slightly different from the token if
        case_sensitive is set to False). Otherwise returns None."""

        if not branch:
            return None
        elif case_sensitive:
            return token if token in branch else None
        elif token in branch:
            return token

        if not token.istitle():
            titled = token.title()
            if titled in branch:
                return titled
        if not token.islower():
            lowered = token.lower()
            if lowered in branch:
                return lowered
        if not token.isupper():
            uppered = token.upper()
            if uppered in branch:
                return uppered

        return None

    def __contains__(self, tokens: List[str]) -> bool:
        """Returns whether the list of tokens are contained in the trie
        (in case-sensitive mode)"""

        return self.contains(tokens)

    def contains(self, tokens: List[str], case_sensitive=True) -> bool:
        """Returns whether the list of tokens are contained in the trie"""

        edges = self.start
        is_terminal = False
        for token in tokens:
            match = self._find_match(token, edges, case_sensitive)
            if not match:
                return False
            edges, is_terminal = edges[token]
        return is_terminal

    def add(self, tokens: List[str]):
        """Adds a new (tokens, value) pair to the trie"""

        # We add new edges to the trie
        edges = self.start
        for token in tokens[:-1]:

            # We create a sub-dictionary if it does not exist
            if token not in edges:
                newdict = {}
                edges[token] = (newdict, False)
                edges = newdict

            else:
                next_edges, is_terminal = edges[token]

                # If the current set of edges is None, map to a dictionary
                if next_edges is None:
                    newdict = {}
                    edges[token] = (newdict, is_terminal)
                    edges = newdict
                else:
                    edges = next_edges

        last_token = tokens[-1]
        if last_token not in edges:
            edges[last_token] = (None, True)
        else:
            edges[last_token] = (edges[last_token][0], True)

        self.len += 1

    def __len__(self) -> int:
        """Returns the total number of (tokens, value) pairs in the trie"""
        return self.len

    def __iter__(self):
        """Generates all elements from the trie"""

        for tokens in self._iter_from_edges(self.start):
            yield tokens

    def _iter_from_edges(self, edges):
        """Generates all elements from a branch in the trie"""

        for token, (sub_branch, is_terminal) in edges.items():
            if is_terminal:
                yield [token]
            if sub_branch is not None:
                for tokens2 in self._iter_from_edges(sub_branch):
                    yield [token, *tokens2]

    def __repr__(self) -> str:
        """Returns a representation of the trie as a flattened list"""

        return list(self).__repr__()


############################################
# Utility functions
############################################


def extract_json_data(json_file: str, cutoff: Optional[int] = None,
                      spacy_model="en_core_web_md") -> Dict[str, Trie]:
    """Extract entities from a Json file and build trie from it (one per class).

    If cutoff is set to a number, stops the extraction after a number of values
    for each class (useful for debugging purposes)."""

    print("Extracting data from", json_file)
    tries = {}
    tokeniser = None
    if json_file.endswith(".json.gz"):
        fd = gzip.open(json_file, "r")
        data = json.loads(fd.read().decode("utf-8"))
        fd.close()
    elif json_file.endswith(".json"):
        fd = open(json_file)
        data = json.load(fd)
        fd.close()
    else:
        raise RuntimeError(str(json_file) + " does not look like a JSON file")    

    for neClass, names in data.items():

        remaining = []
        if cutoff is not None:
            names = names[:cutoff]
        print("Populating trie for class %s (number: %i)" %
                (neClass, len(names)))

        trie = Trie()
        for name in names:
            if type(name) == str:
                tokens = name.split(" ")

                # If the tokens contain special characters, we need to run spacy to
                # ensure we get the same tokenisation as in spacy-tokenised texts
                if any(tok for tok in tokens if not tok.isalpha()
                        and not tok.isnumeric() and not re.match("[A-Z]\\.$", tok)):
                    import spacy
                    tokeniser = tokeniser or spacy.load(
                        spacy_model).tokenizer
                    tokens = [t.text for t in tokeniser(name)]

                if len(tokens) > 0:
                    trie.add(tokens)

            # If the items are already tokenised, we can load the trie faster
            elif type(name) == list:
                if len(name) > 0:
                    trie.add(name)

        tries[neClass] = trie
    return tries
