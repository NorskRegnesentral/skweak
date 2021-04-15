
import skweak
import re

             
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
    assert skweak.utils.get_spans(doc, ["time"]) == {(4,5):"TIME"}
    assert skweak.utils.get_spans(doc, ["number"]) == {(9,11):"PERCENT", (14,15):"CARDINAL"}

def test_gap_tokens(nlp):
    doc = nlp("The Norwegian Computing Center's Employee Union is a long entity, much longer than Jean-Pierre.")
    annotator1 = skweak.heuristics.TokenConstraintAnnotator("test1", skweak.utils.is_likely_proper, "ENT")
    doc = annotator1(doc)
    assert skweak.utils.get_spans(doc, ["test1"]) == {(1,4):"ENT", (5,7):"ENT", (15,18):"ENT"}
    annotator2 = skweak.heuristics.TokenConstraintAnnotator("test2", skweak.utils.is_likely_proper, "ENT")
    annotator2.add_gap_tokens(["'s", "-"])
    doc = annotator2(doc)
    assert skweak.utils.get_spans(doc, ["test2"]) == {(1,7):"ENT", (15,18):"ENT"}

def test_span_annotator(nlp):
    doc = nlp("My name is Pierre Lison and I work at the Norwegian Computing Center.")
    annotator = skweak.heuristics.TokenConstraintAnnotator("proper", skweak.utils.is_likely_proper, "ENT")
    doc = annotator(doc)
    assert skweak.utils.get_spans(doc, ["proper"]) == {(3,5):"ENT", (10, 13): "ENT"}
    annotator2 = skweak.heuristics.SpanConstraintAnnotator("rare_proper", "proper", skweak.utils.is_infrequent)
    doc = annotator2(doc)
    assert skweak.utils.get_spans(doc, ["rare_proper"]) == {(3,5):"ENT"}
    
    
def test_vicinity(nlp):
    doc = nlp("My name is Pierre Lison.")
    annotator1 = skweak.heuristics.TokenConstraintAnnotator("proper", skweak.utils.is_likely_proper, "ENT")
    annotator2 = skweak.heuristics.VicinityAnnotator("neighbours", {"name":"PERSON"}, "proper")
    annotator = skweak.base.CombinedAnnotator().add_annotators(annotator1, annotator2)
    doc = annotator(doc)
    assert skweak.utils.get_spans(doc, ["proper"]) == {(3,5): "ENT"}
    assert skweak.utils.get_spans(doc, ["neighbours"]) == {(3,5): "PERSON"}
    
    
    
def test_model(nlp):
    doc = nlp("My name is Pierre Lison, I live in Oslo and I work at the Norwegian Computing Center.")
    
    annotator = skweak.spacy.ModelAnnotator("core_web_md", "en_core_web_md")
    doc = annotator(doc)
    assert skweak.utils.get_spans(doc, ["core_web_md"]) == {(3,5): "PERSON", (9,10):"GPE", (14,18):"FAC"}
   
    doc.ents = ()
    doc, *_ = annotator.pipe([doc])
    assert skweak.utils.get_spans(doc, ["core_web_md"]) == {(3,5): "PERSON", (9,10):"GPE", (14,18):"FAC"}
    
    doc.ents = ()
    annotator1 = skweak.heuristics.TokenConstraintAnnotator("proper", skweak.utils.is_likely_proper, "ENT")
    annotator2 = skweak.heuristics.VicinityAnnotator("neighbours", {"name":"PERSON"}, "proper")
    annotator = skweak.base.CombinedAnnotator().add_annotators(annotator, annotator1, annotator2)
    doc, *_ = annotator.pipe([doc])
    assert skweak.utils.get_spans(doc, ["core_web_md"]) == {(3,5): "PERSON", (9,10):"GPE", (14,18):"FAC"}
    assert skweak.utils.get_spans(doc, ["proper"]) == {(3,5): "ENT", (9,10):"ENT", (15,18):"ENT"}
