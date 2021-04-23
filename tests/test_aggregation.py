
from skweak.base import CombinedAnnotator
from skweak.spacy import ModelAnnotator
from skweak import utils, aggregation, gazetteers, doclevel, heuristics
from spacy.tokens import Span #type: ignore
import re, json, os
import pytest 
import pandas

@pytest.fixture(scope="session")
def doc(nlp):
    spacy_doc = nlp(re.sub("\\s+", " ", """This is a test for Pierre Lison from the
                     Norwegian Computing Center. Pierre is living in Oslo."""))
    spacy_doc.spans["name"] = [Span(spacy_doc, 5, 7, label="PERSON"),
                               Span(spacy_doc, 13, 14, label="PERSON")]    
    spacy_doc.spans["org"] = [Span(spacy_doc, 9, 12, label="ORG")]
    spacy_doc.spans["place"] = [Span(spacy_doc, 9, 10, label="NORP"),
                                Span(spacy_doc, 17, 18, label="GPE")]
    return spacy_doc  

@pytest.fixture(scope="session")
def doc2(nlp):
    news_text  = """
    ATLANTA  (Reuters) - Retailer Best Buy Co, seeking new ways to appeal to cost-conscious shoppers, said on Tuesday it is selling refurbished 
    versions of Apple Inc's iPhone 3G at its stores that are priced about $50 less than new iPhones. 
    The electronics chain said the used iPhones, which were returned within 30 days of purchase, are priced at $149 for the model with 8 gigabytes of storage, 
    while the 16-gigabyte version is $249. A two-year service contract with AT&T Inc is required. New iPhone 3Gs currently sell for $199 and $299 at 
    Best Buy Mobile stores. "This is focusing on customers' needs, trying to provide as wide a range of products and networks for our consumers," said 
    Scott Moore, vice president of marketing for Best Buy Mobile. Buyers of first-generation iPhones can also upgrade to the faster refurbished 3G models at 
    Best Buy, he said. Moore said AT&T, the exclusive wireless provider for the iPhone, offers refurbished iPhones online. The sale of used iPhones comes as 
    Best Buy, the top consumer electronics chain, seeks ways to fend off increased competition from discounters such as Wal-Mart Stores Inc, which began 
    selling the popular phone late last month. Wal-Mart sells a new 8-gigabyte iPhone 3G for $197 and $297 for the 16-gigabyte model. The iPhone is also 
    sold at Apple stores and AT&T stores. Moore said Best Buy's move was not in response to other retailers' actions. (Reporting by  Karen Jacobs ; Editing 
    by  Andre Grenon )"""
    news_text = re.sub('\\s+', ' ', news_text)
    spacy_doc = nlp(news_text)
    return spacy_doc


@pytest.fixture(scope="session")
def combi_annotator():
    full_annotator = CombinedAnnotator()
    full_annotator.add_annotator(ModelAnnotator("spacy", "en_core_web_md"))
    geo_tries = gazetteers.extract_json_data("data/geonames.json")          
    products_tries = gazetteers.extract_json_data("data/products.json")
    full_annotator.add_annotator(gazetteers.GazetteerAnnotator("geo_cased", geo_tries))
    full_annotator.add_annotator(gazetteers.GazetteerAnnotator("geo_uncased", geo_tries, case_sensitive=False))
    full_annotator.add_annotator(gazetteers.GazetteerAnnotator("products_cased", products_tries))
    full_annotator.add_annotator(gazetteers.GazetteerAnnotator("products_uncased", products_tries, 
                                                               case_sensitive=False))
    full_annotator.add_annotator(heuristics.TokenConstraintAnnotator("proper2_detector", 
                                                                     utils.is_likely_proper, "ENT"))
    full_annotator.add_annotator(heuristics.SpanConstraintAnnotator("full_name_detector", 
                                                                    "proper2_detector", FullNameDetector(), "PERSON"))
    maj_voter = aggregation.MajorityVoter("maj_voter", ["PERSON", "GPE", "ORG", "PRODUCT"],
                                          initial_weights={"doc_history":0, "doc_majority":0})
    full_annotator.add_annotator(maj_voter)
    full_annotator.add_annotator(doclevel.DocumentHistoryAnnotator("doc_history_cased", "maj_voter", ["PERSON", "ORG"]))
    full_annotator.add_annotator(doclevel.DocumentHistoryAnnotator("doc_history_uncased", "maj_voter", ["PERSON", "ORG"],
                                                                   case_sentitive=False))
    full_annotator.add_annotator(doclevel.DocumentMajorityAnnotator("doc_majority_cased", "maj_voter"))
    full_annotator.add_annotator(doclevel.DocumentMajorityAnnotator("doc_majority_uncased", "maj_voter", 
                                                                    case_sensitive=False))
    return full_annotator

  
def test_extract_array(doc):
    
    labels = ["O"] + ["%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"]
    label_df = pandas.DataFrame(utils.spans_to_array(doc, labels=labels), columns=["name", "org", "place"])
    assert label_df.shape == (19, 3)
    assert label_df.apply(lambda x: (x>0).sum()).sum() == 8
    assert (label_df["name"] > 0).sum() == 3
    assert label_df["name"][5]==13    
    assert label_df["name"][6]==15
    assert label_df["place"][17]==4


def test_extract_array2(doc):
    
    labels = ["O"] + ["%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BI"]
    label_df = pandas.DataFrame(utils.spans_to_array(doc, labels=labels), columns=["name", "org", "place"])
   
    assert label_df.shape == (19, 3)
    assert label_df.apply(lambda x: (x>0).sum()).sum() == 8
    assert (label_df["name"] > 0).sum() == 3
    assert label_df["name"][5]==7    
    assert label_df["name"][6]==8
    assert label_df["place"][17]==1
  
    
def test_spans(doc):
    print(doc)
    print(doc.spans)
    for encoding in ["IO", "BIO", "BILUO"]:
        aggregator = aggregation.BaseAggregator("", ["GPE", "NORP", "ORG", "PERSON"], prefixes=encoding)
        obs  = aggregator.get_observation_df(doc)
        print(obs)
        for source in ["name", "org", "place"]:
            spans = utils.token_array_to_spans(obs[source].values, aggregator.out_labels)
            spans = [Span(doc, start, end, label=label) for (start,end),label in spans.items()]
            all_spans = utils.get_spans(doc, [source])
            
            assert spans == all_spans


    
def test_mv(doc):
    mv = aggregation.MajorityVoter("mv", ["GPE", "NORP", "ORG", "PERSON"])
    doc = mv(doc)
    token_labels = doc.spans["mv"].attrs["probs"]
    assert round(sum([prob for probs in token_labels.values() for prob in probs.values()])) == 6
    assert len(token_labels[9]) == 2
    assert ({label for labels in token_labels.values() for label in labels} == 
            {'B-GPE', 'B-NORP', 'B-ORG', 'I-ORG', 'B-PERSON', 'I-PERSON'})

    
def test_mv2(doc):
    mv = aggregation.MajorityVoter("mv", ["GPE", "NORP", "ORG", "PERSON"])
    doc.spans["underspec"] = [Span(doc, 5,7, "ENT"), Span(doc, 9, 12, "ENT")]
    mv.add_underspecified_label("ENT", {"PERSON", "ORG"})
    doc = mv(doc)
    token_labels = doc.spans["mv"].attrs["probs"]
    assert round(sum([prob for probs in token_labels.values() for prob in probs.values()])) == 6
    assert len(token_labels[9]) == 2
    assert abs(token_labels[9]["B-ORG"] - 0.66666) < 0.01
    assert ({label for labels in token_labels.values() for label in labels} == 
            {'B-GPE', 'B-NORP', 'B-ORG', 'I-ORG', 'B-PERSON', 'I-PERSON'})


def test_mv4(doc):
    mv = aggregation.MajorityVoter("mv", ["GPE", "NORP", "ORG", "PERSON"])
    doc.spans["underspec"] = [Span(doc, 5,7, "ENT"), Span(doc, 9, 12, "ENT")]
    mv.add_underspecified_label("ENT", {"PERSON", "ORG"})
    doc = mv(doc)
    token_labels = doc.spans["mv"].attrs["probs"]
    assert round(sum([prob for probs in token_labels.values() for prob in probs.values()])) == 6

def test_hmm(doc):
    hmm = aggregation.HMM("hmm", ["GPE", "NORP", "ORG", "PERSON"], 
                          initial_weights={"underspec":0})
    hmm.fit([doc]*100)
    doc = hmm(doc)
    token_labels = doc.spans["hmm"].attrs["probs"]
    assert round(sum([prob for probs in token_labels.values() for prob in probs.values()])) == 7.0
    assert len(doc.spans["hmm"]) == 4
    assert len(token_labels[9]) == 2
    assert token_labels[9]["B-ORG"] > 3* token_labels[9]["B-NORP"]
    assert token_labels[9]["B-NORP"] > 1E-100
    assert ({label for labels in token_labels.values() for label in labels} == 
            {'B-GPE', 'B-NORP', 'B-ORG', 'I-ORG', 'B-PERSON', 'I-PERSON'})
    

def test_hmm2(doc):
    hmm = aggregation.HMM("hmm", ["GPE", "NORP", "ORG", "PERSON"])
    doc.spans["underspec"] = [Span(doc, 5,7, "ENT"), Span(doc, 9, 12, "ENT")]
    hmm.add_underspecified_label("ENT", {"PERSON", "ORG"})
    hmm.fit([doc]*300)
    doc = hmm(doc)
    token_labels = doc.spans["hmm"].attrs["probs"]
    assert round(sum([prob for probs in token_labels.values() for prob in probs.values()])) == 7
    assert token_labels[9]["B-ORG"] > 0.97
    assert ({label for labels in token_labels.values() for label in labels} == 
            {'B-GPE', 'B-ORG', 'I-ORG', 'B-PERSON', 'B-NORP', 'I-PERSON'})
    


def test_combi(doc2, combi_annotator):
    
    combi_annotator(doc2)
    assert len(doc2.spans["spacy"]) > 35
    assert len(doc2.spans["spacy"]) < 45
    assert len(doc2.spans["geo_cased"]) == 0
    assert len(doc2.spans["geo_uncased"]) == 1
    assert len(doc2.spans["products_cased"]) == 9
    assert len(doc2.spans["proper2_detector"]) >= 32
    assert len(doc2.spans["proper2_detector"]) < 40
    assert len(doc2.spans["full_name_detector"]) in {3,4}
    assert len(doc2.spans["doc_history_cased"]) in {11,12}
    assert len(doc2.spans["doc_majority_cased"]) in {24}
      
def test_hmm3(doc2, combi_annotator):
    hmm = aggregation.HMM("hmm", ["GPE", "PRODUCT", "MONEY", "PERSON", "ORG", "DATE"])
    hmm.add_underspecified_label("ENT", {"GPE", "PRODUCT", "MONEY", "PERSON", "ORG", "DATE"})
    combi_annotator(doc2)
    hmm.fit([doc2]*100)
    doc2 = hmm(doc2)
    assert len(doc2.spans["hmm"]) > 30
    assert len(doc2.spans["hmm"]) < 45
    assert Span(doc2, 1,2, "GPE") in doc2.spans["hmm"]
    found_entities = {(span.text, span.label_) for span in doc2.spans["hmm"]}
    assert ('Scott Moore', 'PERSON') in found_entities
    assert (('197', 'MONEY') in found_entities or ("$197", "MONEY") in found_entities)
    assert ('iPhone 3Gs', 'PRODUCT') in found_entities
 #   assert ('$50', 'MONEY') in found_entities


def test_classification(nlp):
    doc = nlp("A B C")
    doc.spans["lf1"] = [Span(doc, 0, 3, label="C1")]
    doc.spans["lf2"] = [Span(doc, 0, 3, label="C2")]
    doc.spans["lf3"] = []
    doc.spans["lf4"] = [Span(doc, 0, 3, label="C1")]
    mv = aggregation.MajorityVoter("mv", ["C1", "C2"], sequence_labelling=False)
    assert mv.observed_labels == ["O", "C1", "C2"]
    assert mv.out_labels == ["C1", "C2"]
    assert Span(doc, 0, 3, "C1") in mv(doc).spans["mv"]
    assert abs(mv(doc).spans["mv"].attrs["probs"][(0,3)]["C2"] - 1/3) < 0.01
    hmm = aggregation.HMM("hmm", ["C1", "C2"], sequence_labelling=False)
    hmm.fit([doc])
    assert hmm.observed_labels == ["O", "C1", "C2"]
    assert hmm.out_labels == ["C1", "C2"]
    assert Span(doc, 0, 3, "C1") in hmm(doc).spans["hmm"]

def test_classification2(nlp):
    doc = nlp("A B C")
    doc.spans["lf1"] = [Span(doc, 0, 1, label="C1"), 
                        Span(doc, 1, 3, label="C1")]
    doc.spans["lf2"] = [Span(doc, 0, 1, label="C2"), 
                        Span(doc, 1, 2, label="C2")]
    doc.spans["lf3"] = [Span(doc, 0, 2, label="C1"), 
                        Span(doc, 1, 3, label="C1")]
    doc.spans["lf4"] = [Span(doc, 1, 3, label="C2"),
                        Span(doc, 0, 1, label="C1")]
    doc.spans["lf5"] = [Span(doc, 0, 2, label="C1"),
                        Span(doc, 1, 3, label="C1"),
                        Span(doc, 0, 1, label="C1")]
    mv = aggregation.MajorityVoter("mv", ["C1", "C2"], sequence_labelling=False)
    assert mv.observed_labels == ["O", "C1", "C2"]
    assert mv.out_labels == ["C1", "C2"]
    assert Span(doc, 0, 1, "C1") in mv(doc).spans["mv"]
    assert Span(doc, 1, 3, "C1") in mv(doc).spans["mv"]
    assert Span(doc, 1, 2, "C2") in mv(doc).spans["mv"]
    assert Span(doc, 0, 2, "C1") in mv(doc).spans["mv"]
    hmm = aggregation.HMM("hmm", ["C1", "C2"], sequence_labelling=False)
    hmm.fit([doc])
    assert hmm.observed_labels == ["O", "C1", "C2"]
    assert hmm.out_labels == ["C1", "C2"]
    assert Span(doc, 0, 1, "C1") in hmm(doc).spans["hmm"]
    assert Span(doc, 1, 3, "C1") in hmm(doc).spans["hmm"]
    assert Span(doc, 1, 2, "C2") in hmm(doc).spans["hmm"]
    assert Span(doc, 0, 2, "C1") in hmm(doc).spans["hmm"] 
    
def test_classification3(nlp):
    doc = nlp("A B C")
    doc.spans["lf1"] = [Span(doc, 0, 3, label="C1")]
    doc.spans["lf3"] = [Span(doc, 0, 3, label="C3")]
    doc.spans["lf4"] = [Span(doc, 0, 3, label="NOT_C3")]
    mv = aggregation.MajorityVoter("mv", ["C1", "C2", "C3"], sequence_labelling=False)
    mv.add_underspecified_label("NOT_C3", {"C1", "C2"})
    assert mv.observed_labels == ["O", "C1", "C2", "C3", "NOT_C3"]
    assert mv.out_labels == ["C1", "C2", "C3"]
    assert Span(doc, 0, 3, "C1") in mv(doc).spans["mv"]
    assert abs(mv(doc).spans["mv"].attrs["probs"][(0,3)]["C3"] - 1/3) < 0.01
    hmm = aggregation.HMM("hmm", ["C1", "C2", "C3"], sequence_labelling=False)
    hmm.add_underspecified_label("NOT_C3", {"C1", "C2"})
    hmm.fit([doc])
    assert hmm.observed_labels == ["O", "C1", "C2", "C3", "NOT_C3"]
    assert hmm.out_labels == ["C1", "C2", "C3"]
    assert Span(doc, 0, 3, "C1") in hmm(doc).spans["mv"]
    
    
def test_emptydoc(nlp):
    def money_detector(doc):
        for tok in doc[1:]:
            if tok.text[0].isdigit() and tok.nbor(-1).is_currency:
                yield tok.i-1, tok.i+1, "MONEY"
    lf1 = heuristics.FunctionAnnotator("money", money_detector)
    constraint =  lambda tok: re.match("(19|20)\\d{2}$", tok.text)
    lf2= heuristics.TokenConstraintAnnotator("years",constraint, "DATE") #type: ignore
    NAMES = [("Barack", "Obama"), ("Donald", "Trump"), ("Joe", "Biden")]
    trie = gazetteers.Trie(NAMES)
    lf3 = gazetteers.GazetteerAnnotator("presidents", {"PERSON":trie})
    doc = nlp("Donald Trump paid $750 in federal income taxes in 2016")
    doc = lf3(lf2(lf1(doc)))
    doc2 = nlp("And again: Donald Trump paid $750 in federal income taxes in 2016")
    hmm = aggregation.HMM("hmm", ["PERSON", "DATE", "MONEY"])
    hmm.fit_and_aggregate([doc, doc2])   
    assert len(doc.spans["hmm"])==3
    assert "hmm" not in doc2.spans 
               
 
class FullNameDetector():
    """Search for occurrences of full person names (first name followed by at least one title token)"""
   
    def __init__(self):
        fd = open("data/first_names.json")
        self.first_names = set(json.load(fd))
        fd.close()
        
    def __call__(self, span: Span) -> bool:
        
        # We assume full names are between 2 and 5 tokens
        if len(span) < 2 or len(span) > 5:
            return False
        
        return (span[0].text in self.first_names and 
                span[-1].is_alpha and span[-1].is_title)
        
        
        
