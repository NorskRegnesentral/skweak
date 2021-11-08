
import skweak
import re
from spacy.tokens import Span #type: ignore
             
def time_generator(doc):        
    i = 0
    while i < len(doc):
        tok = doc[i]

        if (i < len(doc)-1 and tok.text[0].isdigit() and 
            doc[i+1].lower_ in {"am", "pm", "a.m.", "p.m.", "am.", "pm."}):
            yield i, i+2, "TIME"
            i += 1
        elif tok.text[0].isdigit() and re.match("\\d{1,2}\\:\\d{1,2}", tok.text):
            yield i, i+1, "TIME"
            i += 1
        i += 1  
           
def number_generator(doc):
    i = 0
    while i < len(doc):
        tok = doc[i]
            
        if re.search("\\d", tok.text):
            j = i+1
            if j < len(doc) and doc[j].lower_ in ["%", "percent", "pc.", "pc", "pct", 
                                                  "pct.", "percents", "percentage"]:
                j += 1
                yield i, j, "PERCENT"        
            elif not re.search("[a-zA-Z]", tok.text):
                yield i, j,  "CARDINAL"
            i = j-1
        i += 1

def test_function(nlp):
    doc = nlp("I woke up at 07:30 this morning, being 95% reloaded, with 8 hours of sleep.")
    annotator1 = skweak.heuristics.FunctionAnnotator("time", time_generator)
    annotator2 = skweak.heuristics.FunctionAnnotator("number", number_generator)
    annotator2.add_incompatible_sources(["time"])
    annotator = skweak.base.CombinedAnnotator()
    annotator.add_annotator(annotator1)
    annotator.add_annotator(annotator2)
    doc = annotator(doc)
    assert Span(doc, 4,5, "TIME") in doc.spans["time"]
    assert Span(doc, 9, 11, "PERCENT") in doc.spans["number"]
    assert Span(doc, 14, 15, "CARDINAL") in doc.spans["number"]


def test_gap_tokens(nlp):
    doc = nlp("The Norwegian Computing Center's Employee Union is a long entity, much longer than Jean-Pierre.")
    annotator1 = skweak.heuristics.TokenConstraintAnnotator("test1", skweak.utils.is_likely_proper, "ENT")
    doc = annotator1(doc)
    assert Span(doc, 1, 4, "ENT") in doc.spans["test1"]
    assert Span(doc, 5, 7, "ENT") in doc.spans["test1"]
    assert Span(doc, 15, 18, "ENT") in doc.spans["test1"]
    annotator2 = skweak.heuristics.TokenConstraintAnnotator("test2", skweak.utils.is_likely_proper, "ENT")
    annotator2.add_gap_tokens(["'s", "-"])
    doc = annotator2(doc)
    assert Span(doc, 1, 7, "ENT") in doc.spans["test2"]
    assert Span(doc, 15, 18, "ENT") in doc.spans["test2"]

def test_span_annotator(nlp):
    doc = nlp("My name is Pierre Lison and I work at the Norwegian Computing Center.")
    annotator = skweak.heuristics.TokenConstraintAnnotator("proper", skweak.utils.is_likely_proper, "ENT")
    doc = annotator(doc)
    assert Span(doc, 3, 5, "ENT") in doc.spans["proper"]
    assert Span(doc, 10, 13, "ENT") in doc.spans["proper"]
    annotator2 = skweak.heuristics.SpanConstraintAnnotator("rare_proper", "proper", skweak.utils.is_infrequent)
    doc = annotator2(doc)
    assert Span(doc, 3, 5, "ENT") in doc.spans["rare_proper"]
    
    
def test_vicinity(nlp):
    doc = nlp("My name is Pierre Lison.")
    annotator1 = skweak.heuristics.TokenConstraintAnnotator("proper", skweak.utils.is_likely_proper, "ENT")
    annotator2 = skweak.heuristics.VicinityAnnotator("neighbours", {"name":"PERSON"}, "proper")
    annotator = skweak.base.CombinedAnnotator().add_annotators(annotator1, annotator2)
    doc = annotator(doc)
    assert Span(doc, 3, 5, "ENT") in doc.spans["proper"]
    assert Span(doc, 3, 5, "PERSON") in doc.spans["neighbours"]

    
    
    
def test_model(nlp):
    doc = nlp("My name is Pierre Lison, I live in Oslo and I work at the Norwegian Computing Center.")
    
    annotator = skweak.spacy.ModelAnnotator("core_web_md", "en_core_web_md")
    doc = annotator(doc)
    assert Span(doc, 3, 5, "PERSON") in doc.spans["core_web_md"]
    assert Span(doc, 9, 10, "GPE") in doc.spans["core_web_md"]
    assert (Span(doc, 14, 18, "FAC") in doc.spans["core_web_md"]
            or Span(doc, 14, 18, "ORG") in doc.spans["core_web_md"])
   
    doc.ents = ()
    doc, *_ = annotator.pipe([doc])
    assert Span(doc, 3, 5, "PERSON") in doc.spans["core_web_md"]
    assert Span(doc, 9, 10, "GPE") in doc.spans["core_web_md"]
    assert (Span(doc, 14, 18, "FAC") in doc.spans["core_web_md"]
            or Span(doc, 14, 18, "ORG") in doc.spans["core_web_md"])
    
    doc.ents = ()
    annotator1 = skweak.heuristics.TokenConstraintAnnotator("proper", skweak.utils.is_likely_proper, "ENT")
    annotator2 = skweak.heuristics.VicinityAnnotator("neighbours", {"name":"PERSON"}, "proper")
    annotator = skweak.base.CombinedAnnotator().add_annotators(annotator, annotator1, annotator2)
    doc, *_ = annotator.pipe([doc])
    assert Span(doc, 3, 5, "PERSON") in doc.spans["core_web_md"]
    assert Span(doc, 9, 10, "GPE") in doc.spans["core_web_md"]
    assert (Span(doc, 14, 18, "FAC") in doc.spans["core_web_md"]
            or Span(doc, 14, 18, "ORG") in doc.spans["core_web_md"])
    assert Span(doc, 3, 5, "ENT") in doc.spans["proper"]
    assert Span(doc, 9, 10, "ENT") in doc.spans["proper"]
    assert Span(doc, 15, 18, "ENT") in doc.spans["proper"]
