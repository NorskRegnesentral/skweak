import pytest
from skweak import utils
import os
from spacy.tokens import Span #type: ignore

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
    doc.spans["source1"] = [Span(doc, 0, 2, label="LABEL1"),
                            Span(doc, 4, 5, label="LABEL2")]
    doc.spans["source2"] = [Span(doc, 0, 1, label="LABEL3"),
                            Span(doc, 2, 6, label="LABEL2")]
    doc.spans["source4"] = [Span(doc, 0, 2, label="LABEL2")]
    doc.spans["source3"] = [Span(doc, 7, 9, label="LABEL2"),
                            Span(doc, 1, 4, label="LABEL1")]
             
    assert set((span.start, span.end) for span in 
               utils.get_spans(doc, ["source1", "source2"]))  == {(0,2), (2,6)}                   
    assert set((span.start, span.end) for span in 
               utils.get_spans(doc, ["source1", "source3"])) == {(1,4), (4,5), (7,9)}
    assert {(span.start, span.end):span.label_ for span in 
            utils.get_spans(doc, ["source1", "source4"])}  == {(0,2):"LABEL2", (4,5):"LABEL2"}
    assert set((span.start, span.end) for span in 
               utils.get_spans(doc, ["source2", "source3"])) == {(0,1), (2,6), (7,9)}
    
    
    
    
def test_replace_ner(nlp_small):
    doc = nlp_small("Pierre Lison is working at the Norwegian Computing Center.")
    assert doc.ents[0].text=="Pierre Lison"
    assert doc.ents[0].label_=="PERSON"
    doc.spans["test"] = [Span(doc, 6, 9, label="RESEARCH_ORG")]
    doc = utils.replace_ner_spans(doc, "test")
    assert doc.ents[0].text=="Norwegian Computing Center"
    assert doc.ents[0].label_=="RESEARCH_ORG"


def test_docbins(nlp_small, temp_file="data/temporary_test.docbin"):
    doc = nlp_small("Pierre Lison is working at the Norwegian Computing Center.")
    doc2 = nlp_small("He is working on various NLP topics.")
    doc.spans["test"] = [Span(doc, 0, 2, label="PERSON")]
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
    doc.spans["test"] = [Span(doc, 6, 9, label="RESEARCH_ORG")]
    doc2.spans["test"] = []
    
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
