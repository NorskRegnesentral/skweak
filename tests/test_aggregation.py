
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
    spacy_doc.user_data["spans"] = {"name":{(5,7):"PERSON", (13,14):"PERSON"},
                                    "org":{(9,12):"ORG"}, "place":{(9,10):"NORP", (17,18):"GPE"}}
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
    full_annotator.add_annotator(gazetteers.GazetteerAnnotators("geo", geo_tries))
    full_annotator.add_annotator(gazetteers.GazetteerAnnotators("products", products_tries))
    full_annotator.add_annotator(heuristics.TokenConstraintAnnotator("proper2_detector", 
                                                                     utils.is_likely_proper, "ENT"))
    full_annotator.add_annotator(heuristics.SpanConstraintAnnotator("full_name_detector", 
                                                                    "proper2_detector", FullNameDetector(), "PERSON"))
    maj_voter = aggregation.MajorityVoter("maj_voter", ["PERSON", "GPE", "ORG", "PRODUCT"])
    maj_voter.sources_to_avoid.extend({"doc_history", "doc_majority"})
    full_annotator.add_annotator(maj_voter)
    full_annotator.add_annotator(doclevel.DocumentHistoryAnnotator("doc_history", "maj_voter", ["PERSON", "ORG"]))
    full_annotator.add_annotator(doclevel.DocumentMajorityAnnotator("doc_majority", "maj_voter"))
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
        aggregator = aggregation.BaseAggregator("", ["GPE", "NORP", "ORG", "PERSON"], prefixes=encoding)
        obs  = aggregator.get_observation_df(doc)
        for source in ["name", "org", "place"]:
            spans = utils.token_array_to_spans(obs[source].values, aggregator.out_labels)
            all_spans = utils.get_spans(doc, [source])
            assert spans == all_spans


    
def test_mv(doc):
    mv = aggregation.MajorityVoter("mv", ["GPE", "NORP", "ORG", "PERSON"])
    doc = mv(doc)
    token_labels = doc.user_data["agg_probs"]["mv"]
    assert round(sum([prob for probs in token_labels.values() for prob in probs.values()])) == 6
    assert len(token_labels[9]) == 2
    assert ({label for labels in token_labels.values() for label in labels} == 
            {'B-GPE', 'B-NORP', 'B-ORG', 'I-ORG', 'B-PERSON', 'I-PERSON'})

    
def test_mv2(doc):
    mv = aggregation.MajorityVoter("mv", ["GPE", "NORP", "ORG", "PERSON"])
    doc.user_data["spans"]["underspec"] = {(5,7):"ENT", (9,12):"ENT"}
    mv.add_underspecified_label("ENT", {"PERSON", "ORG"})
    doc = mv(doc)
    token_labels = doc.user_data["agg_probs"]["mv"]
    assert round(sum([prob for probs in token_labels.values() for prob in probs.values()])) == 6
    assert len(token_labels[9]) == 2
    assert abs(token_labels[9]["B-ORG"] - 0.66666) < 0.01
    assert ({label for labels in token_labels.values() for label in labels} == 
            {'B-GPE', 'B-NORP', 'B-ORG', 'I-ORG', 'B-PERSON', 'I-PERSON'})


def test_mv4(doc):
    mv = aggregation.MajorityVoter("mv", ["GPE", "NORP", "ORG", "PERSON"])
    doc.user_data["spans"]["underspec"] = {(5,7):"ENT", (9,12):"ENT"}
    mv.add_underspecified_label("ENT", {"PERSON", "ORG"})
    doc = mv(doc)
    token_labels = doc.user_data["agg_probs"]["mv"]
    assert round(sum([prob for probs in token_labels.values() for prob in probs.values()])) == 6

def test_hmm(doc):
    hmm = aggregation.HMM("hmm", ["GPE", "NORP", "ORG", "PERSON"])
    hmm.sources_to_avoid.append("underspec")
    utils.docbin_writer([doc]*10, "data/test_tmp0.docbin")
    hmm.fit("data/test_tmp0.docbin")
    doc = hmm(doc)
    token_labels = doc.user_data["agg_probs"]["hmm"]
    assert round(sum([prob for probs in token_labels.values() for prob in probs.values()])) == 7.0
    assert len(doc.user_data["agg_spans"]["hmm"]) == 4
    assert len(token_labels[9]) == 2
    assert token_labels[9]["B-ORG"] > 3* token_labels[9]["B-NORP"]
    assert token_labels[9]["B-NORP"] > 1E-100
    assert ({label for labels in token_labels.values() for label in labels} == 
            {'B-GPE', 'B-NORP', 'B-ORG', 'I-ORG', 'B-PERSON', 'I-PERSON'})
    os.remove("data/test_tmp0.docbin")
    

def test_hmm2(doc):
    hmm = aggregation.HMM("hmm", ["GPE", "NORP", "ORG", "PERSON"])
    doc.user_data["spans"]["underspec"] = {(5,7):"ENT", (9,12):"ENT"}
    hmm.add_underspecified_label("ENT", {"PERSON", "ORG"})
    utils.docbin_writer([doc]*100, "data/test_tmp1.docbin")
    hmm.fit("data/test_tmp1.docbin")
    doc = hmm(doc)
    token_labels = doc.user_data["agg_probs"]["hmm"]
    assert round(sum([prob for probs in token_labels.values() for prob in probs.values()])) == 7
    assert token_labels[9]["B-ORG"] > 0.97
    assert ({label for labels in token_labels.values() for label in labels} == 
            {'B-GPE', 'B-ORG', 'I-ORG', 'B-PERSON', 'B-NORP', 'I-PERSON'})
    os.remove("data/test_tmp1.docbin")
    


def test_combi(doc2, combi_annotator):
    
    combi_annotator(doc2)
    assert len(doc2.user_data["spans"]["spacy"]) > 35
    assert len(doc2.user_data["spans"]["spacy"]) < 45
    assert len(doc2.user_data["spans"]["geo_gpe_cased"]) == 0
    assert len(doc2.user_data["spans"]["geo_gpe_uncased"]) == 1
    assert len(doc2.user_data["spans"]["products_product_cased"]) == 9
    assert len(doc2.user_data["spans"]["proper2_detector"]) >= 32
    assert len(doc2.user_data["spans"]["proper2_detector"]) < 40
    assert len(doc2.user_data["spans"]["full_name_detector"]) in {3,4}
    assert len(doc2.user_data["spans"]["doc_history_person_cased"]) in {2,3}
    assert len(doc2.user_data["spans"]["doc_majority_person_cased"]) == 2
    assert len(doc2.user_data["spans"]["doc_majority_org_cased"]) >= 4
    assert len(doc2.user_data["spans"]["doc_majority_product_cased"]) >= 8
      
def test_hmm3(doc2, combi_annotator):
    hmm = aggregation.HMM("hmm", ["GPE", "PRODUCT", "MONEY", "PERSON", "ORG", "DATE"])
    hmm.add_underspecified_label("ENT", {"GPE", "PRODUCT", "MONEY", "PERSON", "ORG", "DATE"})
    combi_annotator(doc2)
    utils.docbin_writer([doc2]*100, "data/test_tmp2.docbin")
    hmm.fit("data/test_tmp2.docbin")
    doc2 = hmm(doc2)
    assert len(doc2.user_data["agg_spans"]["hmm"]) > 30
    assert len(doc2.user_data["agg_spans"]["hmm"]) < 45
    assert doc2.user_data["agg_spans"]["hmm"][(1,2)] == "GPE"
    found_entities = {(doc2[s:e].text, l) for (s,e), l in utils.get_spans(doc2, ["hmm"]).items()}
    assert ('Scott Moore', 'PERSON') in found_entities
    assert (('197', 'MONEY') in found_entities or ("$197", "MONEY") in found_entities)
    assert ('iPhone 3Gs', 'PRODUCT') in found_entities
 #   assert ('$50', 'MONEY') in found_entities
    os.remove("data/test_tmp2.docbin")

    
      
 
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