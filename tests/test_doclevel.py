
import skweak
import re

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
    annotator2 = skweak.doclevel.DocumentHistoryAnnotator("hist", "spacy", ["PERSON", "ORG"])
    doc = annotator2(annotator1(doc))
    assert doc.user_data["spans"]["spacy"][(5, 7)]=='PERSON'
    assert doc.user_data["spans"]["spacy"][(11, 13)] =='ORG'
    assert doc.user_data["spans"]["hist_person_cased"] == {(26, 27): 'PERSON'}
    assert doc.user_data["spans"]["hist_org_cased"] == {(32, 33): 'ORG'}
    assert doc.user_data["spans"]["hist_org_uncased"] == {(32, 33): 'ORG',
                                                          (45, 46): 'ORG'}
    
    
def test_majority(nlp):
    text = re.sub("\\s+", " ", """This is a story about Pierre Lison from Belgium.  He
                  is working as a researcher at the Norwegian Computing Center. The work 
                  of Pierre Lison includes among other weak supervision. He was born and
                  studied in belgium but does not live in Belgium anymore. """)
    doc = nlp(text)
    annotator1 = skweak.spacy.ModelAnnotator("spacy", "en_core_web_md")
    annotator2 = skweak.doclevel.DocumentMajorityAnnotator("maj", "spacy")
    doc = annotator2(annotator1(doc))
    assert doc.user_data["spans"]["spacy"][(5, 7)]=='PERSON'
    assert doc.user_data["spans"]["spacy"][(8, 9)] =='GPE'
    assert doc.user_data["spans"]["spacy"][(17, 21)]=='ORG'
    assert doc.user_data["spans"]["spacy"][(25, 27)]=='PERSON'
    assert doc.user_data["spans"]["spacy"][(45, 46)]=='GPE'
    assert doc.user_data["spans"]["maj_person_cased"] == {(5, 7): 'PERSON', 
                                                          (25, 27): 'PERSON'}
    assert doc.user_data["spans"]["maj_gpe_cased"] == {(8, 9): 'GPE', 
                                                       (45, 46): 'GPE'}
    assert doc.user_data["spans"]["maj_gpe_uncased"] == {(8, 9): 'GPE', 
                                                         (39, 40): 'GPE', 
                                                         (45, 46): 'GPE'}

def test_truecase(nlp):
    text = re.sub("\\s+", " ", """This is A STORY about Pierre LISON from BELGIUM. He IS 
                  WORKING as a RESEARCHER at the Norwegian COMPUTING Center. The WORK of 
                  Pierre LISON includes AMONG OTHER weak SUPERVISION. He WAS BORN AND 
                  studied in belgium BUT does NOT LIVE IN BELGIUM anymore.""")
    doc = nlp(text)
    annotator1 = skweak.spacy.TruecaseAnnotator("truecase", "en_core_web_sm", "data/form_frequencies.json")
    doc = annotator1(doc)
    assert doc.user_data["spans"]["truecase"][(5, 7)] == 'PERSON'
    assert doc.user_data["spans"]["truecase"][(8, 9)] =='GPE'
    assert doc.user_data["spans"]["truecase"][(18, 19)] =='NORP'
    assert doc.user_data["spans"]["truecase"][(25, 27)] =='PERSON'
    assert doc.user_data["spans"]["truecase"][(45, 46)] =='GPE'  

        