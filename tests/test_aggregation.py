
from skweak.base import CombinedAnnotator
from skweak.spacy import ModelAnnotator
from skweak import utils, aggregation, gazetteers, doclevel, heuristics, generative, voting
from spacy.tokens import Span #type: ignore
import re, json, os
import pytest 
import pandas
import skweak

FIRST_NAMES_FILE = os.path.dirname(os.path.dirname(skweak.__file__)) + "/data/first_names.json"
GEONAMES_FILE = os.path.dirname(os.path.dirname(skweak.__file__)) + "/data/geonames.json"
PRODUCTS_FILE = os.path.dirname(os.path.dirname(skweak.__file__)) + "/data/products.json"

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
    geo_tries = gazetteers.extract_json_data(GEONAMES_FILE)          
    products_tries = gazetteers.extract_json_data(PRODUCTS_FILE)
    full_annotator.add_annotator(gazetteers.GazetteerAnnotator("geo_cased", geo_tries))
    full_annotator.add_annotator(gazetteers.GazetteerAnnotator("geo_uncased", geo_tries, case_sensitive=False))
    full_annotator.add_annotator(gazetteers.GazetteerAnnotator("products_cased", products_tries))
    full_annotator.add_annotator(gazetteers.GazetteerAnnotator("products_uncased", products_tries, 
                                                               case_sensitive=False))
    full_annotator.add_annotator(heuristics.TokenConstraintAnnotator("proper2_detector", 
                                                                     utils.is_likely_proper, "ENT"))
    full_annotator.add_annotator(heuristics.SpanConstraintAnnotator("full_name_detector", 
                                                                    "proper2_detector", FullNameDetector(), "PERSON"))
    maj_voter = voting.SequentialMajorityVoter("maj_voter", ["PERSON", "GPE", "ORG", "PRODUCT"],
                                          initial_weights={"doc_history_cased":0, "doc_history_uncased":0,
                                                           "doc_majority_cased":0,  "doc_majority_uncased":0})
    full_annotator.add_annotator(maj_voter)
    full_annotator.add_annotator(doclevel.DocumentHistoryAnnotator("doc_history_cased", "maj_voter", ["PERSON", "ORG"]))
    full_annotator.add_annotator(doclevel.DocumentHistoryAnnotator("doc_history_uncased", "maj_voter", ["PERSON", "ORG"],
                                                                   case_sensitive=False))
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
    for encoding in ["IO", "BIO", "BILUO"]:
        aggregator = voting.SequentialMajorityVoter("", ["GPE", "NORP", "ORG", "PERSON"], prefixes=encoding)
        obs  = aggregator.get_observation_df(doc)
        for source in ["name", "org", "place"]:
            spans = utils.token_array_to_spans(obs[source].values, aggregator.out_labels) #type: ignore
            spans = [Span(doc, start, end, label=label) for (start,end),label in spans.items()]
            all_spans = utils.get_spans(doc, [source])
            
            assert spans == all_spans


    
def test_mv(doc):
    mv = voting.SequentialMajorityVoter("mv", ["GPE", "NORP", "ORG", "PERSON"])
    doc = mv(doc)
    token_labels = doc.spans["mv"].attrs["probs"]
    assert round(sum([prob for probs in token_labels.values() for prob in probs.values()])) == 5
    assert len(token_labels[9]) == 2
    assert ({label for labels in token_labels.values() for label in labels} == 
            {'B-GPE', 'B-NORP', 'B-ORG', 'I-ORG', 'B-PERSON', 'I-PERSON'})

    
def test_mv2(doc):
    mv = voting.SequentialMajorityVoter("mv", ["GPE", "NORP", "ORG", "PERSON"])
    doc.spans["underspec"] = [Span(doc, 5,7, "ENT"), Span(doc, 9, 12, "ENT")]
    mv.add_label_group("ENT", {"PERSON", "ORG"})
    doc = mv(doc)
    token_labels = doc.spans["mv"].attrs["probs"]
    assert round(sum([prob for probs in token_labels.values() for prob in probs.values()])) == 6
    assert len(token_labels[9]) == 3
    assert abs(token_labels[9]["B-ORG"] - 0.5) < 0.01
    assert ({label for labels in token_labels.values() for label in labels} == 
            {'B-GPE', 'B-NORP', 'B-ORG', 'I-ORG', 'B-PERSON', 'I-PERSON'})


def test_mv4(doc):
    mv = voting.SequentialMajorityVoter("mv", ["GPE", "NORP", "ORG", "PERSON"])
    doc.spans["underspec"] = [Span(doc, 5,7, "ENT"), Span(doc, 9, 12, "ENT")]
    mv.add_label_group("ENT", {"PERSON", "ORG"})
    doc = mv(doc)
    token_labels = doc.spans["mv"].attrs["probs"]
    assert round(sum([prob for probs in token_labels.values() for prob in probs.values()])) == 6

def test_hmm(doc):
    hmm = generative.HMM("hmm", ["GPE", "NORP", "ORG", "PERSON"], 
                          initial_weights={"underspec":0})
    hmm.fit([doc]*100, n_iter=1)
    doc = hmm(doc)
    token_labels = doc.spans["hmm"].attrs["probs"]
    assert round(sum([prob for probs in token_labels.values() for prob in probs.values()])) == 7.0
    assert len(doc.spans["hmm"]) == 4
#    assert len(token_labels[9]) == 2
    assert token_labels[9]["B-ORG"] > 3* token_labels[9].get("B-NORP",0)
#    assert token_labels[9]["B-NORP"] > 1E-100
    assert ({label for labels in token_labels.values() for label in labels} == 
            {'B-GPE',  'B-ORG', 'I-ORG', 'B-PERSON', 'I-PERSON'}) # 'B-NORP',
    

def test_hmm2(doc):
    hmm = generative.HMM("hmm", ["GPE", "NORP", "ORG", "PERSON"])
    doc.spans["underspec"] = [Span(doc, 5,7, "ENT"), Span(doc, 9, 12, "ENT")]
    hmm.add_label_group("ENT", {"PERSON", "ORG"})
    hmm.fit([doc]*300)
    doc = hmm(doc)
    token_labels = doc.spans["hmm"].attrs["probs"]
    assert round(sum([prob for probs in token_labels.values() for prob in probs.values()])) == 7
    assert token_labels[9]["B-ORG"] > 0.97
    assert ({label for labels in token_labels.values() for label in labels} == 
            {'B-GPE', 'B-ORG', 'I-ORG', 'B-PERSON', 'I-PERSON'}) # 'B-NORP',
    


def test_combi(doc2, combi_annotator):
    
    combi_annotator(doc2)
    assert len(doc2.spans["spacy"]) > 35
    assert len(doc2.spans["spacy"]) < 45
    assert len(doc2.spans["geo_cased"]) in {0,1}
    assert len(doc2.spans["geo_uncased"]) in {1,2}
    assert len(doc2.spans["products_cased"]) == 9
    assert len(doc2.spans["proper2_detector"]) >= 32
    assert len(doc2.spans["proper2_detector"]) < 40
    assert len(doc2.spans["full_name_detector"]) in {3,4}
    assert len(doc2.spans["doc_history_cased"]) in {11,12}
    assert len(doc2.spans["doc_majority_cased"]) in {23,26}
      
def test_hmm3(doc2, combi_annotator):
    hmm = generative.HMM("hmm", ["GPE", "PRODUCT", "MONEY", "PERSON", "ORG", "DATE"])
    hmm.add_label_group("ENT", {"GPE", "PRODUCT", "PERSON", "ORG", "DATE"})
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
    mv = voting.MajorityVoter("mv", ["C1", "C2"])
    assert mv.observed_labels == ["C1", "C2"]
    assert mv.out_labels == ["C1", "C2"]
    assert Span(doc, 0, 3, "C1") in mv(doc).spans["mv"]
    assert abs(mv(doc).spans["mv"].attrs["probs"][(0,3)]["C2"] - 1/3) < 0.01
    nb = generative.NaiveBayes("nb", ["C1", "C2"])
    nb.fit([doc])
    assert nb.observed_labels == ["C1", "C2"]
    assert nb.out_labels == ["C1", "C2"]
    assert Span(doc, 0, 3, "C1") in nb(doc).spans["nb"]

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
    mv = voting.MajorityVoter("mv", ["C1", "C2"])
    assert mv.observed_labels == ["C1", "C2"]
    assert mv.out_labels == ["C1", "C2"]
    assert Span(doc, 0, 1, "C1") in mv(doc).spans["mv"]
    assert Span(doc, 1, 3, "C1") in mv(doc).spans["mv"]
    assert Span(doc, 1, 2, "C2") in mv(doc).spans["mv"]
    assert Span(doc, 0, 2, "C1") in mv(doc).spans["mv"]
    nb = generative.NaiveBayes("nb", ["C1", "C2"])
    nb.fit([doc])
    assert nb.observed_labels == ["C1", "C2"]
    assert nb.out_labels == ["C1", "C2"]
    assert Span(doc, 0, 1, "C1") in nb(doc).spans["nb"]
    assert Span(doc, 1, 3, "C1") in nb(doc).spans["nb"]
    assert Span(doc, 1, 2, "C2") in nb(doc).spans["nb"]
    assert Span(doc, 0, 2, "C1") in nb(doc).spans["nb"] 
    
def test_classification3(nlp):
    doc = nlp("A B C")
    doc.spans["lf1"] = [Span(doc, 0, 3, label="C1")]
    doc.spans["lf3"] = [Span(doc, 0, 3, label="C3")]
    doc.spans["lf4"] = [Span(doc, 0, 3, label="NOT_C3")]
    mv = voting.MajorityVoter("mv", ["C1", "C2", "C3"])
    mv.add_label_group("NOT_C3", {"C1", "C2"})
    assert mv.observed_labels == ["C1", "C2", "C3", "NOT_C3"]
    assert mv.out_labels == ["C1", "C2", "C3"]
    assert abs(mv(doc).spans["mv"].attrs["probs"][(0,3)]["C1"] - 1/2) < 0.01
    assert abs(mv(doc).spans["mv"].attrs["probs"][(0,3)]["C3"] - 1/3) < 0.01
    nb = generative.NaiveBayes("nb", ["C1", "C2", "C3"])
    nb.add_label_group("NOT_C3", {"C1", "C2"})
    nb.fit([doc])
    assert nb.observed_labels == ["C1", "C2", "C3", "NOT_C3"]
    assert nb.out_labels == ["C1", "C2", "C3"]
    assert Span(doc, 0, 3, "C1") in nb(doc).spans["nb"]
    
    
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
    hmm = generative.HMM("hmm", ["PERSON", "DATE", "MONEY"])
    hmm.fit_and_aggregate([doc, doc2]*10)   
    assert len(doc.spans["hmm"])==3
    assert len(doc2.spans["hmm"])==0 


def test_multilabel_classifier(nlp):
    doc = nlp("This is a a short text")
    doc.spans["source1"] = [Span(doc, 0, len(doc), "C1")]
    doc.spans["source2"] = [Span(doc, 0, len(doc), "C2")]
    doc.spans["source3"] = [Span(doc, 0, len(doc), "C2")]
    doc.spans["source4"] = []
    
    mv = skweak.voting.MultilabelMajorityVoter("mv", ["C1", "C2"]) 
    doc = mv(doc)
    assert abs(doc.spans["mv"].attrs["probs"][(0,len(doc))]["C1"] - 1.0) <= 0.01
    assert abs(doc.spans["mv"].attrs["probs"][(0,len(doc))]["C2"] - 1.0) <= 0.01
    mv.set_exclusive_labels({"C1", "C2"})
    doc = mv(doc)
    assert abs(doc.spans["mv"].attrs["probs"][(0,len(doc))]["C1"] - 0.33) <= 0.01
    assert abs(doc.spans["mv"].attrs["probs"][(0,len(doc))]["C2"] - 0.66) <= 0.01
    
    nb = skweak.generative.MultilabelNaiveBayes("nb", ["C1", "C2"])   
    nb.fit([doc]*100)  
    doc = nb(doc)
    assert abs(doc.spans["nb"].attrs["probs"][(0,len(doc))]["C1"] - 1.0) <= 0.01
    assert abs(doc.spans["nb"].attrs["probs"][(0,len(doc))]["C2"] - 1.0) <= 0.01
    nb = skweak.generative.MultilabelNaiveBayes("nb", ["C1", "C2"])   
    nb.set_exclusive_labels({"C1", "C2"})
    nb.fit([doc]*100)  
    doc = nb(doc)
    assert abs(doc.spans["nb"].attrs["probs"][(0,len(doc))]["C1"] - 0.27) <= 0.1
    assert abs(doc.spans["nb"].attrs["probs"][(0,len(doc))]["C2"] - 0.72) <= 0.1
    
    
def test_multilabel_mv(nlp):
    
    ann1 = skweak.heuristics.FunctionAnnotator("ann1", lambda d: [(i, i+1, "ORG") 
                                                                  for i in range(len(d)) 
                                                                  if d[i].text=="Norwegian"]
                                               + [(i, i+3, "ORG") for i in range(len(d)-2) 
                                                  if d[i:i+3].text=="Bill Gates Foundation"]) 
    ann2 = skweak.heuristics.FunctionAnnotator("ann2", lambda d: [(i, i+1, "NORP") 
                                                                  for i in range(len(d)) 
                                                                  if d[i].text=="Norwegian"] 
                                               +  [(i, i+2, "PERSON") for i in range(len(d)-1) 
                                                   if d[i:i+2].text=="Bill Gates"]) #type: ignore
    ann3 = skweak.heuristics.FunctionAnnotator("ann3", lambda d: [(i, i+1, "NUM") 
                                                                  for i in range(len(d)) 
                                                                  if re.match("\d+", d[i].text)])  
    ann4 = skweak.heuristics.FunctionAnnotator("ann4", lambda d:[(ent.start, ent.end, "NUM") 
                                                                 for ent in d.ents 
                                                                 if ent.label_=="CARDINAL"]) 
    mv = skweak.voting.MultilabelSequentialMajorityVoter("mv", ["GPE", "NORP", "ORG", "PERSON", "NUM"]) 
    doc = nlp("This is a test for Norwegian and the Bill Gates Foundation, with a number 34.")
    doc = mv(ann4(ann3(ann2(ann1(doc))))) 
   
    assert abs(doc.spans["mv"].attrs["probs"][5]["B-ORG"] - 0.75) <= 0.01
    assert abs(doc.spans["mv"].attrs["probs"][5]["B-NORP"] - 0.75) <= 0.01
    assert abs(doc.spans["mv"].attrs["probs"][8]["B-ORG"] - 0.75) <= 0.01
    assert abs(doc.spans["mv"].attrs["probs"][8]["B-PERSON"] - 0.75) <= 0.01
    assert abs(doc.spans["mv"].attrs["probs"][9]["I-ORG"] - 0.75) <= 0.01
    assert abs(doc.spans["mv"].attrs["probs"][9]["I-PERSON"] - 0.75) <= 0.01
    assert abs(doc.spans["mv"].attrs["probs"][10]["I-ORG"] - 0.75) <= 0.01
    assert abs(doc.spans["mv"].attrs["probs"][15]["B-NUM"] - 0.947) <= 0.01


def test_multilabel_mv2(nlp):
    
    ann1 = skweak.heuristics.FunctionAnnotator("ann1", lambda d: [(i, i+1, "ORG") 
                                                                  for i in range(len(d)) 
                                                                  if d[i].text=="Norwegian"]
                                               + [(i, i+3, "ORG") for i in range(len(d)-2) 
                                                  if d[i:i+3].text=="Bill Gates Foundation"]) 
    ann2 = skweak.heuristics.FunctionAnnotator("ann2", lambda d: [(i, i+1, "NORP") 
                                                                  for i in range(len(d)) 
                                                                  if d[i].text=="Norwegian"] 
                                               +  [(i, i+2, "PERSON") for i in range(len(d)-1) 
                                                   if d[i:i+2].text=="Bill Gates"]) #type: ignore
    ann3 = skweak.heuristics.FunctionAnnotator("ann3", lambda d: [(i, i+1, "NUM") 
                                                                  for i in range(len(d)) 
                                                                  if re.match("\d+", d[i].text)])  
    ann4 = skweak.heuristics.FunctionAnnotator("ann4", lambda d:[(ent.start, ent.end, "NUM") 
                                                                 for ent in d.ents 
                                                                 if ent.label_=="CARDINAL"]) 
    mv = skweak.voting.MultilabelSequentialMajorityVoter("mv", ["GPE", "NORP", "ORG", "PERSON", "NUM"]) 
    mv.set_exclusive_labels({"ORG", "PERSON"})
    doc = nlp("This is a test for Norwegian and the Bill Gates Foundation, with a number 34.")
    doc = mv(ann4(ann3(ann2(ann1(doc))))) 
   
    assert abs(doc.spans["mv"].attrs["probs"][5]["B-ORG"] - 0.75) <= 0.01
    assert abs(doc.spans["mv"].attrs["probs"][5]["B-NORP"] - 0.75) <= 0.01
    assert abs(doc.spans["mv"].attrs["probs"][8]["B-ORG"] - 0.474) <= 0.01
    assert abs(doc.spans["mv"].attrs["probs"][8]["B-PERSON"] - 0.474) <= 0.01
    assert abs(doc.spans["mv"].attrs["probs"][9]["I-ORG"] - 0.474) <= 0.01
    assert abs(doc.spans["mv"].attrs["probs"][9]["I-PERSON"] - 0.474) <= 0.01
    assert abs(doc.spans["mv"].attrs["probs"][10]["I-ORG"] - 0.75) <= 0.01
    assert abs(doc.spans["mv"].attrs["probs"][15]["B-NUM"] - 0.947) <= 0.01   


def test_multilabel_hmm(nlp):
    
    ann1 = skweak.heuristics.FunctionAnnotator("ann1", lambda d: [(i, i+1, "ORG") 
                                                                  for i in range(len(d)) 
                                                                  if d[i].text=="Norwegian"]
                                               + [(i, i+3, "ORG") for i in range(len(d)-2) 
                                                  if d[i:i+3].text=="Bill Gates Foundation"]) 
    ann2 = skweak.heuristics.FunctionAnnotator("ann2", lambda d: [(i, i+1, "NORP") 
                                                                  for i in range(len(d)) 
                                                                  if d[i].text=="Norwegian"] 
                                               +  [(i, i+2, "PERSON") for i in range(len(d)-1) 
                                                   if d[i:i+2].text=="Bill Gates"]) #type: ignore
    ann3 = skweak.heuristics.FunctionAnnotator("ann3", lambda d: [(i, i+1, "NUM") 
                                                                  for i in range(len(d)) 
                                                                  if re.match("\d+", d[i].text)])  
    ann4 = skweak.heuristics.FunctionAnnotator("ann4", lambda d:[(ent.start, ent.end, "NUM") 
                                                                 for ent in d.ents 
                                                                 if ent.label_=="CARDINAL"]) 
    
    hmm = skweak.generative.MultilabelHMM("hmm", ["GPE", "NORP", "ORG", "PERSON", "NUM"]) 
    doc = nlp("This is a test for Norwegian and the Bill Gates Foundation, with a number 34.")
    doc = ann4(ann3(ann2(ann1(doc))))
    hmm.fit([doc]*100)
    doc = hmm(doc)
    
    assert abs(doc.spans["hmm"].attrs["probs"][5]["B-ORG"] - 0.999) <= 0.01
    assert abs(doc.spans["hmm"].attrs["probs"][5]["B-NORP"] - 0.999) <= 0.01
    assert abs(doc.spans["hmm"].attrs["probs"][8]["B-ORG"] - 0.999) <= 0.01
    assert abs(doc.spans["hmm"].attrs["probs"][8]["B-PERSON"] - 0.999) <= 0.01
    assert abs(doc.spans["hmm"].attrs["probs"][9]["I-ORG"] - 0.999) <= 0.01
    assert abs(doc.spans["hmm"].attrs["probs"][9]["I-PERSON"] - 0.999) <= 0.01
    assert abs(doc.spans["hmm"].attrs["probs"][10]["I-ORG"] - 0.999) <= 0.01
    assert abs(doc.spans["hmm"].attrs["probs"][15]["B-NUM"] - 0.999) <= 0.01


def test_multilabel_hmm2(nlp):
    
    ann1 = skweak.heuristics.FunctionAnnotator("ann1", lambda d: [(i, i+1, "ORG") 
                                                                  for i in range(len(d)) 
                                                                  if d[i].text=="Norwegian"]
                                               + [(i, i+3, "ORG") for i in range(len(d)-2) 
                                                  if d[i:i+3].text=="Bill Gates Foundation"]) 
    ann2 = skweak.heuristics.FunctionAnnotator("ann2", lambda d: [(i, i+1, "NORP") 
                                                                  for i in range(len(d)) 
                                                                  if d[i].text=="Norwegian"] 
                                               +  [(i, i+2, "PERSON") for i in range(len(d)-1) 
                                                   if d[i:i+2].text=="Bill Gates"]) #type: ignore
    ann3 = skweak.heuristics.FunctionAnnotator("ann3", lambda d: [(i, i+1, "NUM") 
                                                                  for i in range(len(d)) 
                                                                  if re.match("\d+", d[i].text)])  
    ann4 = skweak.heuristics.FunctionAnnotator("ann4", lambda d:[(ent.start, ent.end, "NUM") 
                                                                 for ent in d.ents 
                                                                 if ent.label_=="CARDINAL"]) 
    hmm = skweak.generative.MultilabelHMM("hmm", ["GPE", "NORP", "ORG", "PERSON", "NUM"]) 
    hmm.set_exclusive_labels({"ORG", "PERSON"})
    doc = nlp("This is a test for Norwegian and the Bill Gates Foundation, with a number 34.")
    doc = ann4(ann3(ann2(ann1(doc))))
    hmm.fit([doc]*100)
    doc = hmm(doc)
   
    assert abs(doc.spans["hmm"].attrs["probs"][5]["B-ORG"] - 0.999) <= 0.01
    assert abs(doc.spans["hmm"].attrs["probs"][5]["B-NORP"] - 0.999) <= 0.01
    assert abs(doc.spans["hmm"].attrs["probs"][8]["B-ORG"] - 0.499) <= 0.01
    assert abs(doc.spans["hmm"].attrs["probs"][8]["B-PERSON"] - 0.499) <= 0.01
    assert abs(doc.spans["hmm"].attrs["probs"][9]["I-ORG"] - 0.499) <= 0.01
    assert abs(doc.spans["hmm"].attrs["probs"][9]["I-PERSON"] - 0.499) <= 0.01
    assert abs(doc.spans["hmm"].attrs["probs"][10]["I-ORG"] - 0.999) <= 0.01
    assert abs(doc.spans["hmm"].attrs["probs"][15]["B-NUM"] - 0.999) <= 0.01   

class FullNameDetector():
    """Search for occurrences of full person names (first name followed by at least one title token)"""
   
    def __init__(self):
        fd = open(FIRST_NAMES_FILE)
        self.first_names = set(json.load(fd))
        fd.close()
        
    def __call__(self, span: Span) -> bool:
        
        # We assume full names are between 2 and 5 tokens
        if len(span) < 2 or len(span) > 5:
            return False
        
        return (span[0].text in self.first_names and 
                span[-1].is_alpha and span[-1].is_title)
        
        
        
