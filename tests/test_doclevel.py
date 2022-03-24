
import skweak
import re
from spacy.tokens import Span  # type: ignore

def test_subsequences():
    text = ["This", "is", "a", "test", "."]
    subsequences = [["This"], ["is"], ["a"], ["test"], ["."], ["This", "is"], ["is", "a"], 
                    ["a", "test"], ["test", "."], ["This", "is", "a"], ["is", "a", "test"], 
                    ["a", "test", "."], ["This", "is", "a", "test"], ["is", "a", "test", "."]]
    assert sorted(skweak.utils.get_subsequences(text)) == sorted(subsequences + [text])
    
    
def test_history(nlp):
    text = re.sub("\\s+", " ", """This is a story about Pierre Lison and his work at 
                  Yetanothername Inc., which is just a name we invented. But of course, 
                  Lison did not really work for Yetanothername, because it is a fictious 
                  name, even when spelled like YETANOTHERNAME.""")
    doc = nlp(text)
    annotator1 = skweak.spacy.ModelAnnotator("spacy", "en_core_web_sm")
    annotator2 = skweak.doclevel.DocumentHistoryAnnotator("hist_cased", "spacy", ["PERSON", "ORG"])
    annotator3 = skweak.doclevel.DocumentHistoryAnnotator("hist_uncased", "spacy", ["PERSON", "ORG"],
                                                          case_sensitive=False)
    doc = annotator3(annotator2(annotator1(doc)))
    assert Span(doc, 5, 7, "PERSON") in doc.spans["spacy"]
    assert Span(doc, 11, 13, "ORG") in doc.spans["spacy"]
    assert Span(doc, 26, 27, "PERSON") in doc.spans["hist_cased"]
    assert Span(doc, 32, 33, "ORG") in doc.spans["hist_cased"]
    assert Span(doc, 32, 33, "ORG") in doc.spans["hist_uncased"]
    print("DEBUG", doc[45], doc[45].lemma_, doc[45].tag_)
    assert Span(doc, 45, 46, "ORG") in doc.spans["hist_uncased"]
    
    
def test_majority(nlp):
    text = re.sub("\\s+", " ", """This is a story about Pierre Lison from Belgium.  He
                  is working as a researcher at the Norwegian Computing Center. The work 
                  of Pierre Lison includes among other weak supervision. He was born and
                  studied in belgium but does not live in Belgium anymore. """)
    doc = nlp(text)
    annotator1 = skweak.spacy.ModelAnnotator("spacy", "en_core_web_md")
    annotator2 = skweak.doclevel.DocumentMajorityAnnotator("maj_cased", "spacy")
    annotator3 = skweak.doclevel.DocumentMajorityAnnotator("maj_uncased", "spacy", 
                                                           case_sensitive=False)
    doc = annotator3(annotator2(annotator1(doc)))
    assert Span(doc, 5, 7, "PERSON") in doc.spans["spacy"]
    assert Span(doc, 8, 9, "GPE") in doc.spans["spacy"]
    assert Span(doc, 17, 21, "ORG") in doc.spans["spacy"]
    assert Span(doc, 25, 27, "PERSON") in doc.spans["spacy"]
    assert Span(doc, 45, 46, "GPE") in doc.spans["spacy"]
    assert Span(doc, 5, 7, "PERSON") in doc.spans["maj_cased"]
    assert Span(doc, 25, 27, "PERSON") in doc.spans["maj_cased"]
    assert Span(doc, 8, 9, "GPE") in doc.spans["maj_cased"]
    assert Span(doc, 45, 46, "GPE") in doc.spans["maj_cased"]
    assert Span(doc, 8, 9, "GPE") in doc.spans["maj_uncased"]
 #   assert Span(doc, 39, 40, "GPE") in doc.spans["maj_uncased"]
    assert Span(doc, 45, 46, "GPE") in doc.spans["maj_uncased"]


def test_truecase(nlp):
    text = re.sub("\\s+", " ", """This is A STORY about Pierre LISON from BELGIUM. He IS 
                  WORKING as a RESEARCHER at the Norwegian COMPUTING Center. The WORK of 
                  Pierre LISON includes AMONG OTHER weak SUPERVISION. He WAS BORN AND 
                  studied in belgium BUT does NOT LIVE IN BELGIUM anymore.""")
    doc = nlp(text)
    annotator1 = skweak.spacy.TruecaseAnnotator("truecase", "en_core_web_sm", "data/form_frequencies.json")
    doc = annotator1(doc)
    assert Span(doc, 5, 7, "PERSON") in doc.spans["truecase"]
    assert Span(doc, 8, 9, "GPE") in doc.spans["truecase"]
    assert Span(doc, 18, 19, "NORP") in doc.spans["truecase"]
    assert Span(doc, 25, 27, "PERSON") in doc.spans["truecase"]
    assert Span(doc, 45, 46, "GPE") in doc.spans["truecase"]        
