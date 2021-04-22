from skweak import gazetteers, utils
import json, gzip
from spacy.tokens import Span #type: ignore

def test_trie1():
    trie = gazetteers.Trie()
    trie.add(["Donald", "Trump"])
    trie.add(["Donald", "Duck"])
    trie.add(["Donald", "Duck", "Magazine"])
    
    assert ["Donald", "Trump"] in trie
    assert ["Donald", "Duck"] in trie 
    assert ["Donald", "Duck", "Magazine"] in trie 
    assert ["Donald"] not in trie 
    assert ["Trump"] not in trie 
    assert ["Pierre"] not in trie
    assert trie.find_longest_match(["Donald", "Trump", "was", "the"]) == ["Donald", "Trump"]
    assert trie.find_longest_match(["Donald", "Duck", "was", "the"]) == ["Donald", "Duck"]
    assert trie.find_longest_match(["Donald", "Duck", "Magazine", "the"]) == ["Donald", "Duck", "Magazine"]
    
    assert trie.find_longest_match(["Donald"]) == []
    assert trie.find_longest_match(["Pierre"]) == []
    
    assert sorted(trie) == [["Donald", "Duck"], ["Donald", "Duck", "Magazine"], 
                            ["Donald", "Trump"]]
    
    
def test_trie2(nlp, json_file="data/wikidata_small_tokenised.json.gz", cutoff=100):
    tries = gazetteers.extract_json_data(json_file, cutoff=cutoff)
    fd = gzip.open(json_file, "r")
    data = json.loads(fd.read().decode("utf-8"))
    fd.close()
    
    for neClass, names_for_class in data.items():
        nb_names = 0
        trie = tries[neClass]
        for name in names_for_class:
            tokens = list(name)
            if len(tokens)==0:
                continue
            assert tokens in trie
            assert trie.find_longest_match(tokens) == tokens
            nb_names += 1
            if nb_names >= cutoff:
                break   
    
def test_trie_case_insensitive():
    trie = gazetteers.Trie()
    trie.add(["Donald", "Trump"])
    trie.add(["Donald", "Duck"])
    trie.add(["Donald", "Duck", "Magazine"])
    
    assert trie.find_longest_match(["Donald", "Trump", "was", "the"], 
                               case_sensitive=False) == ["Donald", "Trump"]
    assert trie.find_longest_match(["Donald", "trump", "was", "the"], 
                               case_sensitive=False) == ["Donald", "Trump"]
    assert trie.find_longest_match(["DONALD", "trump", "was", "the"], 
                               case_sensitive=False) == ["Donald", "Trump"]
    assert trie.find_longest_match(["Donald", "Duck", "Magazine", "the"], 
                               case_sensitive=False) == ["Donald", "Duck", "Magazine"]
    assert trie.find_longest_match(["Donald", "Duck", "magazine", "the"], 
                               case_sensitive=False) == ["Donald", "Duck", "Magazine"]
    
    assert trie.find_longest_match(["Donald"], case_sensitive=False) == []

def test_gazetteer(nlp):
    trie = gazetteers.Trie()
    trie.add(["Donald", "Trump"])
    trie.add(["Donald", "Duck"])
    trie.add(["Donald", "Duck", "Magazine"])
    trie.add(["Apple"])
   
    gazetteer = gazetteers.GazetteerAnnotator("test_gazetteer", {"ENT":trie})
    doc1 = nlp("Donald Trump is now reading Donald Duck Magazine.")
    doc2 = nlp("Donald Trump (unrelated with Donald Duck) is now reading Donald Duck Magazine.")
    doc1, doc2 = gazetteer.pipe([doc1, doc2])
    assert Span(doc1, 0, 2, "ENT") in doc1.spans["test_gazetteer"]
    assert Span(doc1, 5, 8, "ENT") in doc1.spans["test_gazetteer"]
    assert Span(doc2, 0, 2, "ENT") in doc2.spans["test_gazetteer"]
    assert Span(doc2, 5, 7, "ENT") in doc2.spans["test_gazetteer"]
    assert Span(doc2, 11, 14, "ENT") in doc2.spans["test_gazetteer"]

    gazetteer = gazetteers.GazetteerAnnotator("test_gazetteer", {"ENT":trie}, case_sensitive=False)
    doc1 = nlp("Donald Trump is now reading Donald Duck Magazine.")
    doc2 = nlp("Donald trump (unrelated with donald Duck) is now reading Donald Duck magazine.")

    doc3 = nlp("At Apple, we do not like to simply eat an apple.")
    doc1, doc2, doc3 = gazetteer.pipe([doc1, doc2, doc3])
    assert Span(doc1, 0, 2, "ENT") in doc1.spans["test_gazetteer"]
    assert Span(doc1, 5, 8, "ENT") in doc1.spans["test_gazetteer"]
    assert Span(doc2, 0, 2, "ENT") in doc2.spans["test_gazetteer"]
    assert Span(doc2, 5, 7, "ENT") in doc2.spans["test_gazetteer"]
    assert Span(doc2, 11, 14, "ENT") in doc2.spans["test_gazetteer"]
    assert Span(doc3, 1, 2, "ENT") in doc3.spans["test_gazetteer"]

 
def test_gazetteer2(nlp):
    
    class Trie2(gazetteers.Trie):
        def __init__(self):
            super(Trie2, self).__init__()
            self.nb_queries = 0
            
        def find_longest_match(self, tokens, case_sensitive=True):
            self.nb_queries += 1
            return super(Trie2, self).find_longest_match(tokens, case_sensitive)
            
    trie = Trie2()
    trie.add(["Donald", "Trump"])
    trie.add(["Donald", "Duck"])
    trie.add(["Donald", "Duck", "Magazine"])
    
    gazetteer = gazetteers.GazetteerAnnotator("test_gazetteer", {"ENT":trie})
    doc1 = nlp("Donald Trump is now reading Donald Duck Magazine.")
    gazetteer(doc1)
    assert trie.nb_queries == 5
    