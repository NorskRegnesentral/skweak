import pytest
from skweak import utils
import os


def test_likely_proper(nlp_small, nlp):     
    for nlpx in [nlp_small, nlp]:
        doc = nlpx("This is a test. Please tell me that is works.")
        for tok in doc:
            assert not utils.is_likely_proper(tok)
        doc = nlpx("Pierre Lison is living in Oslo.")
        for i, tok in enumerate(doc):
            assert utils.is_likely_proper(tok) == (i in {0,1,5})
        doc = nlpx("Short sentence. But here, Beyond can be an organisation.")
        for i, tok in enumerate(doc):
            assert utils.is_likely_proper(tok) == (i in {6})
            
    doc = nlp_small("Buying an iPad makes you ekrjøewlkrj in the USA.")
    for i, tok in enumerate(doc):
        assert utils.is_likely_proper(tok) == (i in {2,8})
    doc = nlp("Buying an iPad makes you ekrjøewlkrj in the USA.")
    for i, tok in enumerate(doc):
        assert utils.is_likely_proper(tok) == (i in {2,8,5})

            
def test_infrequent(nlp_small, nlp):
    doc = nlp_small("The Moscow Art Museum awaits you")
    assert not utils.is_infrequent(doc[:5])
    doc = nlp("The Moscow Art Museum awaits you")
    assert utils.is_infrequent(doc[:5])
    doc = nlp_small("completelyUnknownToken")
    assert not utils.is_infrequent(doc[:1])
    doc = nlp("completelyUnknownToken")
    assert utils.is_infrequent(doc[:1])

def test_compound(nlp):
    doc = nlp("The White House focuses on risk assessment.")
    assert not utils.in_compound(doc[0])
    assert utils.in_compound(doc[1])
    assert utils.in_compound(doc[2])
    assert not utils.in_compound(doc[3])
    assert not utils.in_compound(doc[4])
    assert utils.in_compound(doc[5])
    assert utils.in_compound(doc[6])
    assert not utils.in_compound(doc[7])
    


def test_get_spans(nlp_small):
    
    doc = nlp_small("This is just a small test for checking that the method works correctly")
    doc.user_data["spans"] = {"source1":{(0,2):"LABEL1", (4,5):"LABEL2"}, 
                              "source2":{(0,1):"LABEL3", (2,6):"LABEL2"},
                              "source4":{(0,2):"LABEL2"}}
    doc.user_data["agg_spans"] = {"source3":{(7,9):"LABEL2", (1,4):"LABEL1"}}
    doc.user_data["agg_probs"] = {"source3":{7:{"B-LABEL2":0.7}, 8:{"I-LABEL2":0.6}, 
                                                    1:{"B-LABEL1":0.9}, 2:{"I-LABEL1":0.55}, 3:{"I-LABEL1":0.85}}}
                                  
    assert set(utils.get_spans(doc, ["source1", "source2"]).keys()) == {(0,2), (2,6)}
    assert set(utils.get_spans(doc, ["source1", "source3"]).keys()) == {(1,4), (4,5), (7,9)}
    assert utils.get_spans(doc, ["source1", "source4"]) == {(0,2):"LABEL2", (4,5):"LABEL2"}
    assert set(utils.get_spans(doc, ["source2", "source3"]).keys()) == {(0,1), (2,6), (7,9)}
    assert set(utils.get_spans(doc, ["source2", "source4"]).keys()) == {(0,2), (2,6)}
    assert set(utils.get_spans(doc, ["source3", "source4"]).keys()) == {(1,4), (7,9)}
    
    
    
def test_replace_ner(nlp_small):
    doc = nlp_small("Pierre Lison is working at the Norwegian Computing Center.")
    assert doc.ents[0].text=="Pierre Lison"
    assert doc.ents[0].label_=="PERSON"
    doc.user_data["spans"] = {"test":{(6, 9):"RESEARCH_ORG"}}
    doc = utils.replace_ner_spans(doc, "test")
    assert doc.ents[0].text=="Norwegian Computing Center"
    assert doc.ents[0].label_=="RESEARCH_ORG"


def test_docbins(nlp_small, temp_file="data/temporary_test.docbin"):
    doc = nlp_small("Pierre Lison is working at the Norwegian Computing Center.")
    doc2 = nlp_small("He is working on various NLP topics.")
    doc.user_data["spans"] = {"test":{(0,2):"PERSON"}}
    utils.docbin_writer([doc, doc2], temp_file)
    doc3, doc4 = list(utils.docbin_reader(temp_file, "en_core_web_sm"))
    assert doc.text == doc3.text 
    assert doc2.text == doc4.text 
    assert [(e.text, e.label_) for e in doc.ents] == [(e.text, e.label_) for e in doc3.ents]
    assert doc.user_data == doc3.user_data 
    os.remove(temp_file)
    
    

def test_json(nlp_small, temp_file="data/temporary_test.json"):
    import spacy
    if int(spacy.__version__[0]) > 2:
        return
    
    doc = nlp_small("Pierre Lison is working at the Norwegian Computing Center.")
    doc2 = nlp_small("He is working on various NLP topics.")
    doc.user_data["spans"] = {"test":{(6, 9):"RESEARCH_ORG"}}
    doc2.user_data["spans"] = {"test":{}}
    
    utils.json_writer([doc, doc2], temp_file, source="test")
    fd = open(temp_file, "r")
    assert "I-RESEARCH_ORG" in fd.read()
    fd.close()
    os.remove(temp_file)
    
    
def test_valid_transitions():
    assert utils.is_valid_start("O")    
    assert utils.is_valid_start("B-ORG")     
    assert not utils.is_valid_start("I-ORG")     
    assert utils.is_valid_start("I-ORG", "IO")
    assert utils.is_valid_start("U-ORG", "BILUO")    
    assert not utils.is_valid_start("L-ORG")
    
    assert utils.is_valid_transition("O","O")    
    assert utils.is_valid_transition("O","B-ORG")
    assert utils.is_valid_transition("O","U-ORG")
    assert not utils.is_valid_transition("O","I-ORG")
    assert utils.is_valid_transition("O","I-ORG", "IO")
    assert not utils.is_valid_transition("O","L-ORG")
    
    assert utils.is_valid_transition("B-ORG","I-ORG")
    assert utils.is_valid_transition("B-ORG","L-ORG", "BILUO")
    assert not utils.is_valid_transition("B-ORG","I-GPE")
    assert not utils.is_valid_transition("B-ORG","B-ORG", "BILUO")
    assert utils.is_valid_transition("I-ORG", "B-ORG")
    assert not utils.is_valid_transition("I-ORG", "B-ORG", "BILUO")
    assert not utils.is_valid_transition("I-ORG", "O", "BILUO")
    assert utils.is_valid_transition("I-ORG", "O")
    assert utils.is_valid_transition("I-ORG", "O", "IO")
    assert utils.is_valid_transition("I-ORG", "U-GPE")
    assert not utils.is_valid_transition("I-ORG", "I-GPE")
    assert utils.is_valid_transition("I-ORG", "U-GPE")
    assert utils.is_valid_transition("I-ORG", "L-ORG", "BILUO")
    assert not utils.is_valid_transition("L-ORG", "L-ORG", "BILUO")
    assert not utils.is_valid_transition("L-ORG", "I-ORG", "BILUO")
    assert utils.is_valid_transition("U-ORG", "U-ORG")
    assert utils.is_valid_transition("U-ORG", "U-GPE")
    assert utils.is_valid_transition("U-ORG", "O")
    assert utils.is_valid_transition("L-ORG", "O", "BILUO")
    assert not utils.is_valid_transition("I-ORG", "O", "BILUO")
