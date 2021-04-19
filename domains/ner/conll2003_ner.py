from typing import Iterable, Tuple
import re, json, os
import snips_nlu_parsers
from skweak.base import CombinedAnnotator, SpanAnnotator
from skweak.spacy import ModelAnnotator, TruecaseAnnotator
from skweak.heuristics import FunctionAnnotator, TokenConstraintAnnotator, SpanConstraintAnnotator, SpanEditorAnnotator
from skweak.gazetteers import GazetteerAnnotator, extract_json_data
from skweak.doclevel import DocumentHistoryAnnotator, DocumentMajorityAnnotator
from skweak.aggregation import MajorityVoter
from skweak import utils
from spacy.tokens import Doc, Span  # type: ignore
from . import data_utils

# Data files for gazetteers
WIKIDATA = os.path.dirname(__file__) + "/../../data/wikidata_tokenised.json"
WIKIDATA_SMALL = os.path.dirname(__file__) + "/../../data/wikidata_small_tokenised.json"
COMPANY_NAMES = os.path.dirname(__file__) + "/../../data/company_names_tokenised.json"
GEONAMES = os.path.dirname(__file__) + "/../../data/geonames.json"
CRUNCHBASE = os.path.dirname(__file__) + "/../../data/crunchbase.json"
PRODUCTS = os.path.dirname(__file__) + "/../../data/products.json"
FIRST_NAMES = os.path.dirname(__file__) + "/../../data/first_names.json"
FORM_FREQUENCIES = os.path.dirname(__file__) + "/../../data/form_frequencies.json"


############################################
# Combination of all annotators
############################################


class NERAnnotator(CombinedAnnotator):
    """Annotator of entities in documents, combining several sub-annotators (such as gazetteers,
    spacy models etc.). To add all annotators currently implemented, call add_all(). """

    def add_all(self):
        """Adds all implemented annotation functions, models and filters"""

        print("Loading shallow functions")
        self.add_shallow()
        print("Loading Spacy NER models")
        self.add_models()
        print("Loading gazetteer supervision modules")
        self.add_gazetteers()
        print("Loading document-level supervision sources")
        self.add_doc_level()

        return self

    def add_shallow(self):
        """Adds shallow annotation functions"""

        # Detection of dates, time, money, and numbers
        self.add_annotator(FunctionAnnotator("date_detector", date_generator))
        self.add_annotator(FunctionAnnotator("time_detector", time_generator))
        self.add_annotator(FunctionAnnotator("money_detector", money_generator))

        # Detection based on casing
        proper_detector = TokenConstraintAnnotator("proper_detector", utils.is_likely_proper, "ENT")

        # Detection based on casing, but allowing some lowercased tokens
        proper2_detector = TokenConstraintAnnotator("proper2_detector", utils.is_likely_proper, "ENT")
        proper2_detector.add_gap_tokens(data_utils.LOWERCASED_TOKENS | data_utils.NAME_PREFIXES)

        # Detection based on part-of-speech tags
        nnp_detector = TokenConstraintAnnotator("nnp_detector", lambda tok: tok.tag_ in {"NNP", "NNPS"}, "ENT")

        # Detection based on dependency relations (compound phrases)
        compound = lambda tok: utils.is_likely_proper(tok) and utils.in_compound(tok)
        compound_detector = TokenConstraintAnnotator("compound_detector", compound, "ENT")

        exclusives = ["date_detector", "time_detector", "money_detector"]
        for annotator in [proper_detector, proper2_detector, nnp_detector, compound_detector]:
            annotator.add_incompatible_sources(exclusives)
            annotator.add_gap_tokens(["'s", "-"])
            self.add_annotator(annotator)

            # We add one variants for each NE detector, looking at infrequent tokens
            infrequent_name = "infrequent_%s" % annotator.name
            self.add_annotator(SpanConstraintAnnotator(infrequent_name, annotator.name, utils.is_infrequent))

        # Other types (legal references etc.)
        misc_detector = FunctionAnnotator("misc_detector", misc_generator)
        legal_detector = FunctionAnnotator("legal_detector", legal_generator)

        # Detection of companies with a legal type
        ends_with_legal_suffix = lambda x: x[-1].lower_.rstrip(".") in data_utils.LEGAL_SUFFIXES
        company_type_detector = SpanConstraintAnnotator("company_type_detector", "proper2_detector",
                                                        ends_with_legal_suffix, "COMPANY")

        # Detection of full person names
        full_name_detector = SpanConstraintAnnotator("full_name_detector", "proper2_detector",
                                                     FullNameDetector(), "PERSON")

        for annotator in [misc_detector, legal_detector, company_type_detector, full_name_detector]:
            annotator.add_incompatible_sources(exclusives)
            self.add_annotator(annotator)

        # General number detector
        number_detector = FunctionAnnotator("number_detector", number_generator)
        number_detector.add_incompatible_sources(exclusives + ["legal_detector", "company_type_detector"])
        self.add_annotator(number_detector)

        self.add_annotator(SnipsAnnotator("snips"))
        return self

    def add_models(self):
        """Adds Spacy NER models to the annotator"""

        self.add_annotator(ModelAnnotator("core_web_md", "en_core_web_md"))
        self.add_annotator(TruecaseAnnotator("core_web_md_truecase", "en_core_web_md", FORM_FREQUENCIES))
        self.add_annotator(ModelAnnotator("BTC", os.path.dirname(__file__) + "/../../data/btc"))
        self.add_annotator( TruecaseAnnotator("BTC_truecase", os.path.dirname(__file__) + "/../../data/btc", FORM_FREQUENCIES))

        # Avoid spans that start with an article
        editor = lambda span: span[1:] if span[0].lemma_ in {"the", "a", "an"} else span
        self.add_annotator(SpanEditorAnnotator("edited_BTC", "BTC", editor))
        self.add_annotator(SpanEditorAnnotator("edited_BTC_truecase", "BTC_truecase", editor))
        self.add_annotator(SpanEditorAnnotator("edited_core_web_md", "core_web_md", editor))
        self.add_annotator(SpanEditorAnnotator("edited_core_web_md_truecase", "core_web_md_truecase", editor))

        return self

    def add_gazetteers(self, full_load=True):
        """Adds gazetteer supervision models (company names and wikidata)."""

        # Annotation of company names based on a large list of companies
        # company_tries = extract_json_data(COMPANY_NAMES) if full_load else {}

        # Annotation of company, person and location names based on wikidata
        wiki_tries = extract_json_data(WIKIDATA) if full_load else {}

        # Annotation of company, person and location names based on wikidata (only entries with descriptions)
        wiki_small_tries = extract_json_data(WIKIDATA_SMALL)

        # Annotation of location names based on geonames
        geo_tries = extract_json_data(GEONAMES)

        # Annotation of organisation and person names based on crunchbase open data
        crunchbase_tries = extract_json_data(CRUNCHBASE)

        # Annotation of product names
        products_tries = extract_json_data(PRODUCTS)

        exclusives = ["date_detector", "time_detector", "money_detector", "number_detector"]
        for name, tries in {"wiki":wiki_tries, "wiki_small":wiki_small_tries,
                            "geo":geo_tries, "crunchbase":crunchbase_tries, "products":products_tries}.items():
            
            # For each KB, we create two gazetters (case-sensitive or not)
            cased_gazetteer = GazetteerAnnotator("%s_cased"%name, tries, case_sensitive=True)
            uncased_gazetteer = GazetteerAnnotator("%s_uncased"%name, tries, case_sensitive=False)
            cased_gazetteer.add_incompatible_sources(exclusives)
            uncased_gazetteer.add_incompatible_sources(exclusives)
            self.add_annotators(cased_gazetteer, uncased_gazetteer)
                
            # We also add new sources for multitoken entities (which have higher confidence)
            multitoken_cased = SpanConstraintAnnotator("multitoken_%s"%(cased_gazetteer.name), 
                                                       cased_gazetteer.name, lambda s: len(s) > 1)
            multitoken_uncased = SpanConstraintAnnotator("multitoken_%s"%(uncased_gazetteer.name), 
                                                         uncased_gazetteer.name, lambda s: len(s) > 1)
            self.add_annotators(multitoken_cased, multitoken_uncased)
                
        return self

    def add_doc_level(self):
        """Adds document-level supervision sources"""

        self.add_annotator(ConLL2003Standardiser())

        maj_voter = MajorityVoter("doclevel_voter", ["LOC", "MISC", "ORG", "PER"], 
                                  initial_weights={"doc_history":0, "doc_majority":0})
        maj_voter.add_underspecified_label("ENT", {"LOC", "MISC", "ORG", "PER"})     
        self.add_annotator(maj_voter)   
           
        self.add_annotator(DocumentHistoryAnnotator("doc_history_cased", "doclevel_voter", ["PER", "ORG"]))
        self.add_annotator(DocumentHistoryAnnotator("doc_history_uncased", "doclevel_voter", ["PER", "ORG"],
                                                    case_sentitive=False))
        
        maj_voter = MajorityVoter("doclevel_voter", ["LOC", "MISC", "ORG", "PER"],
                                  initial_weights={"doc_majority":0})
        maj_voter.add_underspecified_label("ENT", {"LOC", "MISC", "ORG", "PER"})
        self.add_annotator(maj_voter)

        self.add_annotator(DocumentMajorityAnnotator("doc_majority_cased", "doclevel_voter"))
        self.add_annotator(DocumentMajorityAnnotator("doc_majority_uncased", "doclevel_voter", 
                                                     case_sensitive=False))
        return self


############################################
# Heuristics
############################################


def date_generator(doc):
    """Searches for occurrences of date patterns in text"""

    spans = []

    i = 0
    while i < len(doc):
        tok = doc[i]
        if tok.lemma_ in data_utils.DAYS | data_utils.DAYS_ABBRV:
            spans.append((i, i + 1, "DATE"))
        elif tok.is_digit and re.match("\\d+$", tok.text) and int(tok.text) > 1920 and int(tok.text) < 2040:
            spans.append((i, i + 1, "DATE"))
        elif tok.lemma_ in data_utils.MONTHS | data_utils.MONTHS_ABBRV:
            if tok.tag_ == "MD":  # Skipping "May" used as auxiliary
                pass
            elif i > 0 and re.match("\\d+$", doc[i - 1].text) and int(doc[i - 1].text) < 32:
                spans.append((i - 1, i + 1, "DATE"))
            elif i > 1 and re.match("\\d+(?:st|nd|rd|th)$", doc[i - 2].text) and doc[i - 1].lower_ == "of":
                spans.append((i - 2, i + 1, "DATE"))
            elif i < len(doc) - 1 and re.match("\\d+$", doc[i + 1].text) and int(doc[i + 1].text) < 32:
                spans.append((i, i + 2, "DATE"))
                i += 1
            else:
                spans.append((i, i + 1, "DATE"))
        i += 1

    for start, end, content in utils.merge_contiguous_spans(spans, doc):
        yield start, end, content


def time_generator(doc):
    """Searches for occurrences of time patterns in text"""

    i = 0
    while i < len(doc):
        tok = doc[i]

        if (i < len(doc) - 1 and tok.text[0].isdigit() and
                doc[i + 1].lower_ in {"am", "pm", "a.m.", "p.m.", "am.", "pm."}):
            yield i, i + 2, "TIME"
            i += 1
        elif tok.text[0].isdigit() and re.match("\\d{1,2}\\:\\d{1,2}", tok.text):
            yield i, i + 1, "TIME"
            i += 1
        i += 1


def money_generator(doc):
    """Searches for occurrences of money patterns in text"""

    i = 0
    while i < len(doc):
        tok = doc[i]
        if tok.text[0].isdigit():
            j = i + 1
            while (j < len(doc) and (doc[j].text[0].isdigit() or doc[j].norm_ in data_utils.MAGNITUDES)):
                j += 1

            found_symbol = False
            if i > 0 and doc[i - 1].text in (data_utils.CURRENCY_CODES | data_utils.CURRENCY_SYMBOLS):
                i = i - 1
                found_symbol = True
            if (j < len(doc) and doc[j].text in
                    (data_utils.CURRENCY_CODES | data_utils.CURRENCY_SYMBOLS | {"euros", "cents", "rubles"})):
                j += 1
                found_symbol = True

            if found_symbol:
                yield i, j, "MONEY"
            i = j
        else:
            i += 1


def number_generator(doc):
    """Searches for occurrences of number patterns (cardinal, ordinal, quantity or percent) in text"""

    i = 0
    while i < len(doc):
        tok = doc[i]

        if tok.lower_ in data_utils.ORDINALS:
            yield i, i + 1, "ORDINAL"

        elif re.search("\\d", tok.text):
            j = i + 1
            while (j < len(doc) and (doc[j].norm_ in data_utils.MAGNITUDES)):
                j += 1
            if j < len(doc) and doc[j].lower_.rstrip(".") in data_utils.UNITS:
                j += 1
                yield i, j, "QUANTITY"
            elif j < len(doc) and doc[j].lower_ in ["%", "percent", "pc.", "pc", "pct", "pct.", "percents",
                                                    "percentage"]:
                j += 1
                yield i, j, "PERCENT"
            else:
                yield i, j, "CARDINAL"
            i = j - 1
        i += 1


class FullNameDetector():
    """Search for occurrences of full person names (first name followed by at least one title token)"""

    def __init__(self):
        fd = open(FIRST_NAMES)
        self.first_names = set(json.load(fd))
        fd.close()

    def __call__(self, span: Span) -> bool:
        # We assume full names are between 2 and 5 tokens
        if len(span) < 2 or len(span) > 5:
            return False

        return (span[0].text in self.first_names and
                span[-1].is_alpha and span[-1].is_title)


class SnipsAnnotator(SpanAnnotator):
    """Annotation using the Snips NLU entity parser. 
       You must install  "snips-nlu-parsers" (pip install snips-nlu-parsers) to make it work.
    """
    
    def __init__(self, name: str):
        """Initialise the annotation tool."""

        super(SnipsAnnotator, self).__init__(name)
        self.parser = snips_nlu_parsers.BuiltinEntityParser.build(language="en")

    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:
        """Runs the parser on the spacy document, and convert the result to labels."""

        text = doc.text

        # The current version of Snips has a bug that makes it crash with some rare
        # Turkish characters, or mentions of "billion years"
        text = text.replace("’", "'").replace("”", "\"").replace("“", "\"").replace("—", "-")
        text = text.encode("iso-8859-15", "ignore").decode("iso-8859-15")
        text = re.sub("(\\d+) ([bm]illion(?: (?:\\d+|one|two|three|four|five|six|seven" +
                      "|eight|nine|ten))? years?)", "\\g<1>.0 \\g<2>", text)

        results = self.parser.parse(text)
        for result in results:
            span = doc.char_span(result["range"]["start"], result["range"]["end"])
            if span is None or span.text.lower() in {"now"} or span.text in {"may"}:
                continue
            label = None
            if (result["entity_kind"] == "snips/number" and span.text.lower() not in
                    {"one", "some", "few", "many", "several"}):
                label = "CARDINAL"
            elif (result["entity_kind"] == "snips/ordinal" and span.text.lower() not in
                  {"first", "second", "the first", "the second"}):
                label = "ORDINAL"
            elif result["entity_kind"] == "snips/temperature":
                label = "QUANTITY"
            elif result["entity_kind"] == "snips/amountOfMoney":
                label = "MONEY"
            elif result["entity_kind"] == "snips/percentage":
                label = "PERCENT"
            elif result["entity_kind"] in {"snips/date", "snips/datePeriod", "snips/datetime"}:
                label = "DATE"
            elif result["entity_kind"] in {"snips/time", "snips/timePeriod"}:
                label = "TIME"

            if label:
                yield span.start, span.end, label

def legal_generator(doc):
    legal_spans = []
    for span in utils.get_spans(doc, ["proper2_detector", "nnp_detector"]):
        if not utils.is_likely_proper(doc[span.end-1]):
            continue         
        last_token = doc[span.end-1].text.title().rstrip("s")
                  
        if last_token in data_utils.LEGAL:     
            legal_spans.append((span.start,span.end, "LAW"))
                     
    
    # Handling legal references such as Article 5
    for i in range(len(doc) - 1):
        if doc[i].text.rstrip("s") in {"Article", "Paragraph", "Section", "Chapter", "§"}:
            if doc[i + 1].text[0].isdigit() or doc[i + 1].text in data_utils.ROMAN_NUMERALS:
                start, end = i, i + 2
                if (i < len(doc) - 3 and doc[i + 2].text in {"-", "to", "and"}
                        and (doc[i + 3].text[0].isdigit() or doc[i + 3].text in data_utils.ROMAN_NUMERALS)):
                    end = i + 4
                legal_spans.append((start, end, "LAW"))

    # Merge contiguous spans of legal references ("Article 5, Paragraph 3")
    legal_spans = utils.merge_contiguous_spans(legal_spans, doc)
    for start, end, label in legal_spans:
        yield start, end, label


def misc_generator(doc):
    """Detects occurrences of countries and various less-common entities (NORP, FAC, EVENT, LANG)"""
    
    spans = set(doc.spans["proper2_detector"])
    spans |= {doc[i:i+1] for i in range(len(doc))}
    
    for span in sorted(spans):

        span_text = span.text
        if span_text.isupper():
            span_text = span_text.title()
        last_token = doc[span.end-1].text

        if span_text in data_utils.COUNTRIES:
            yield span.start, span.end, "GPE"

        if len(span) <= 3 and (span in data_utils.NORPS or last_token in data_utils.NORPS 
                               or last_token.rstrip("s") in data_utils.NORPS):
            yield span.start, span.end, "NORP"
    
        if span in data_utils.LANGUAGES and doc[span.start].tag_=="NNP":
            yield span.start, span.end, "LANGUAGE"
            
        if last_token in data_utils.FACILITIES and len(span) > 1:
            yield span.start, span.end, "FAC"     

        if last_token in data_utils.EVENTS  and len(span) > 1:
            yield span.start, span.end, "EVENT"     
    
       
    
############################################
# Standardisation of the output labels
############################################


class ConLL2003Standardiser(SpanAnnotator):
    """Annotator taking existing annotations and standardising them
    to fit the ConLL 2003 tag scheme"""

    def __init__(self):
        super(ConLL2003Standardiser, self).__init__("")

    def __call__(self, doc):
        """Annotates one single document"""     
               
        for source in doc.spans:
               
            new_spans = []  
            for span in doc.spans[source]:
                if "\n" in span.text:
                    continue
                elif span.label_=="PERSON":
                    new_spans.append(Span(doc, span.start, span.end, label="PER"))
                elif span.label_ in {"ORGANIZATION", "ORGANISATION", "COMPANY"}:
                    new_spans.append(Span(doc, span.start, span.end, label="ORG"))
                elif span.label_ in {"GPE"}:
                    new_spans.append(Span(doc, span.start, span.end, label="LOC"))
                elif span.label_ in {"EVENT", "FAC", "LANGUAGE", "LAW", "NORP", "PRODUCT", "WORK_OF_ART"}:
                    new_spans.append(Span(doc, span.start, span.end, label="MISC"))
                else:
                    new_spans.append(span)         
            doc.spans[source] = new_spans      
        return doc

