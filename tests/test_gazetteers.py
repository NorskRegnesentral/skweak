from skweak import gazetteers, utils
import json

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
    
    
def test_trie2(nlp, json_file="data/wikidata_small.json", cutoff=100):
    tries = gazetteers.extract_json_data(json_file, cutoff=cutoff)
    fd = open(json_file)
    data = json.load(fd)
    fd.close()
    
    for neClass, names_for_class in data.items():
        nb_names = 0
        trie = tries[neClass]
        for name in names_for_class:
            tokens = [tok.text for tok in nlp.tokenizer(name.strip())]
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
   
    gazetteer = gazetteers.GazetteerAnnotator("test_gazetteer", trie, "ENT")
    doc1 = nlp("Donald Trump is now reading Donald Duck Magazine.")
    doc2 = nlp("Donald Trump (unrelated with Donald Duck) is now reading Donald Duck Magazine.")
    doc1, doc2 = gazetteer.pipe([doc1, doc2])
    assert utils.get_spans(doc1, ["test_gazetteer"]) == {(0,2): "ENT", (5,8):"ENT"}
    assert utils.get_spans(doc2, ["test_gazetteer"]) == {(0,2): "ENT", (5,7): "ENT", (11, 14): "ENT"}

    gazetteer = gazetteers.GazetteerAnnotator("test_gazetteer", trie, "ENT", case_sensitive=False)
    doc1 = nlp("Donald Trump is now reading Donald Duck Magazine.")
    doc2 = nlp("Donald trump (unrelated with donald Duck) is now reading Donald Duck magazine.")

    doc3 = nlp("At Apple, we do not like to simply eat an apple.")
    doc1, doc2, doc3 = gazetteer.pipe([doc1, doc2, doc3])
    assert utils.get_spans(doc1, ["test_gazetteer"]) == {(0,2): "ENT", (5,8):"ENT"}
    assert utils.get_spans(doc2, ["test_gazetteer"]) == {(0,2): "ENT", (5,7): "ENT", (11, 14): "ENT"}
    assert utils.get_spans(doc3, ["test_gazetteer"]) == {(1,2):"ENT"}

 
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
    
    gazetteer = gazetteers.GazetteerAnnotator("test_gazetteer", trie, "ENT")
    doc1 = nlp("Donald Trump is now reading Donald Duck Magazine.")
    gazetteer(doc1)
    assert trie.nb_queries == 5
    