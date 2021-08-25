
import re

from spacy.tokens import Span #type: ignore
import pytest

from skweak.analysis import LFAnalysis


@pytest.fixture(scope="session")
def analysis_doc(nlp):
    """ Generate a sample document with conflicts, agreements, and overlaps.
    """
    spacy_doc = nlp(
        re.sub("\\s+", " ", """This is a test for Pierre Lison from the
                     Norwegian Computing Center. Pierre is living in Oslo."""))
    spacy_doc.spans["name_1"] = [
        Span(spacy_doc, 5, 7, label="PERSON"),
        Span(spacy_doc, 13, 14, label="PERSON")
    ]
    spacy_doc.spans["name_2"] = [
        Span(spacy_doc, 5, 7, label="PERSON")
    ] 
    spacy_doc.spans["org_1"] = [
        Span(spacy_doc, 9, 12, label="ORG")
    ]
    spacy_doc.spans["org_2"] = [
        Span(spacy_doc, 9, 11, label="ORG"), 
        Span(spacy_doc, 17, 18, label="ORG"),
    ]
    spacy_doc.spans["place_1"] = [
        Span(spacy_doc, 9, 10, label="NORP"),
        Span(spacy_doc, 17, 18, label="GPE")
    ]
    return spacy_doc


# ---------------------
# LABEL OVERLAP TESTS
# ---------------------
def test_overlaps_with_strict_match_with_prefixes(analysis_doc):
    """ Test expected overlaps across below spans:
  
    Spans:
        "name_1": Pierre (B-PERSON), Lison (L-PERSON), Pierre(U-PERSON)
        "name_2": Pierre (B-PERSON), Lison (L-PERSON)
        "org_1": Norwegian (B-ORG), Computing (I-ORG), Center(L-ORG)
        "org_2": Norwegian (B-ORG), Computing (L-ORG), Oslo (U-ORG)
        "place_1": Norwegian (U-NORP), Oslo (U-GPE)

    Overlaps:
        B-PERSON: 1 Overlap / 1 Token = 1.0
        L-PERSON: 1 Overlap / 1 Token = 1.0
        U-PERSON: 0 Overlap / 1 Token = 0.0
        B-ORG: 1 Overlap / 1 Token = 1.0
        I-ORG: 1 Overlap / 1 Token = 1.0
        L-ORG: 1 Overlap / 2 Tokens = 0.5
        U-ORG: 1 Overlap / 1 Token = 1.0
        U-NORP: 1 Overlap / 1 Token = 1.0
        U-GPE: 1 Overlap / 1 Token = 1.0
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=True
    )
    result = lf_analysis.label_overlap()
    assert result['overlap']['B-PERSON'] == 1.0
    assert result['overlap']['L-PERSON'] == 1.0
    assert result['overlap']['U-PERSON'] == 0.0
    assert result['overlap']['B-ORG'] == 1.0
    assert result['overlap']['I-ORG'] == 1.0
    assert result['overlap']['L-ORG'] == 1/2
    assert result['overlap']['U-ORG'] == 1.0
    assert result['overlap']['U-NORP'] == 1.0
    assert result['overlap']['U-GPE'] == 1.0


def test_overlaps_without_strict_match_with_prefixes(analysis_doc):
    """ Test expected overlaps across below spans:
  
    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Overlaps:
        PERSON: 2 Overlaps / 3 Tokens = 0.66
        ORG: 3 Overlaps / 4 Tokens = 0.75
        NORP: 1 Overlap / 1 Token = 1.0
        GPE: 1 Overlap / 1 Token = 1.0
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=False
    )
    result = lf_analysis.label_overlap()
    assert result['overlap']['PERSON'] == 2/3
    assert result['overlap']['ORG'] == 3/4
    assert result['overlap']['NORP'] == 1.0
    assert result['overlap']['GPE'] == 1.0


def test_overlaps_without_strict_match_without_prefixes(analysis_doc):
    """ Test expected overlaps across below spans:
  
    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Overlaps:
        PERSON: 2 Overlaps / 3 Tokens = 0.66
        ORG: 3 Overlaps / 4 Tokens = 0.75
        NORP: 1 Overlap / 1 Token = 1.0
        GPE: 1 Overlap / 1 Token = 1.0
    """
    labels = ["O", "GPE", "NORP", "ORG", "PERSON"]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=False
    )
    result = lf_analysis.label_overlap()
    assert result['overlap']['PERSON'] == 2/3
    assert result['overlap']['ORG'] == 3/4
    assert result['overlap']['NORP'] == 1.0
    assert result['overlap']['GPE'] == 1.0

# ---------------------
# LABEL CONFLICT TESTS
# ---------------------
def test_conflicts_with_strict_match_with_prefixes(analysis_doc):
    """ Test expected conflicts across below spans:
  
    Spans:
        "name_1": Pierre (B-PERSON), Lison (L-PERSON), Pierre(U-PERSON)
        "name_2": Pierre (B-PERSON), Lison (L-PERSON)
        "org_1": Norwegian (B-ORG), Computing (I-ORG), Center(L-ORG)
        "org_2": Norwegian (B-ORG), Computing (L-ORG), Oslo (U-ORG)
        "place_1": Norwegian (U-NORP), Oslo (U-GPE)

    Conflicts:
        B-PERSON: 0 Conflicts / 1 Token = 0.0
        L-PERSON: 0 Conflicts / 1 Token = 0.0
        U-PERSON: 0 Conflicts / 1 Token = 0.0
        B-ORG: 1 Conflict / 1 Token = 1.0
            (1 Conflict: Norwegian - B-ORG/U-NORP)
        I-ORG: 1 Conflict / 1 Token = 1.0
            (1 Conflict: Computing - I-ORG/L-ORG)
        L-ORG: 1 Conflict / 2 Tokens = 0.5
            (1 Conflict: Computing - I-ORG/L-ORG )
        U-ORG: 1 Conflict / 1 Token = 1.0
            (1 Conflict: Oslo - U-ORG/U-GPE)
        U-NORP: 1 Conflict / 1 Token = 1.0
            (1 Conflict: Norwegian - B-ORG/U-NORP)
        U-GPE: 1 Conflict / 1 Token = 1.0
            (1 Conflict: Oslo - U-ORG/U-GPE)
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=True
    )
    result = lf_analysis.label_conflict()
    assert result['conflict']['B-PERSON'] == 0.0
    assert result['conflict']['L-PERSON'] == 0.0
    assert result['conflict']['U-PERSON'] == 0.0
    assert result['conflict']['B-ORG'] == 1.0
    assert result['conflict']['I-ORG'] == 1.0
    assert result['conflict']['L-ORG'] == 0.5
    assert result['conflict']['U-ORG'] == 1.0
    assert result['conflict']['U-NORP'] == 1.0
    assert result['conflict']['U-GPE'] == 1.0


def test_conflicts_without_strict_match_with_prefixes(analysis_doc):
    """ Test expected conflicts across below spans:
  
    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Conflicts:
        PERSON: 0 Conflicts / 3 Tokens = 0.0
        ORG: 2 Conflicts / 4 Tokens = 0.5
            (2 Conflicts: Norwegian - ORG/NORP, Oslo - ORG/GPE)
        NORP: 1 Conflict / 1 Token = 1.0
             (1 Conflict: Norweigan - ORG/NORP)
        GPE: 1 Conflict / 1 Token = 1.0
            (1 Conflict : Oslo - ORG/GPE)
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=False
    )
    result = lf_analysis.label_conflict()
    assert result['conflict']['PERSON'] == 0.0
    assert result['conflict']['ORG'] == 2/4
    assert result['conflict']['NORP'] == 1.0
    assert result['conflict']['GPE'] == 1.0


def test_conflicts_without_strict_match_without_prefixes(analysis_doc):
    """ Test expected conflicts across below spans:
  
    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Conflicts:
        PERSON: 0 Conflicts / 3 Tokens = 0.0
        ORG: 2 Conflicts / 4 Tokens = 0.5
            (2 Conflicts: Norwegian - ORG/NORP, Oslo - ORG/GPE)
        NORP: 1 Conflict / 1 Token = 1.0
             (1 Conflict: Norweigan - ORG/NORP)
        GPE: 1 Conflict / 1 Token = 1.0
            (1 Conflict : Oslo - ORG/GPE)
    """
    labels = ["O", "GPE", "NORP", "ORG", "PERSON"]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=False
    )
    result = lf_analysis.label_conflict()
    assert result['conflict']['PERSON'] == 0.0
    assert result['conflict']['ORG'] == 2/4
    assert result['conflict']['NORP'] == 1.0
    assert result['conflict']['GPE'] == 1.0


# ----------------
# LF TARGETS TESTS
# ----------------
def test_lf_targets(analysis_doc):
    """ Test expected targets across below spans:
    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)
    """
    labels = ["O", "GPE", "NORP", "ORG", "PERSON"]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_target_labels()

    assert result["name_1"] == [lf_analysis.label2idx["PERSON"]]
    assert result["name_2"] == [lf_analysis.label2idx["PERSON"]]
    assert result["org_1"] == [lf_analysis.label2idx["ORG"]]
    assert result["org_2"] == [lf_analysis.label2idx["ORG"]]
    assert set(result["place_1"]) == set([
        lf_analysis.label2idx["NORP"],
        lf_analysis.label2idx["GPE"]])


# ---------------------------
# LF COVERAGE TESTS w/out agg
# ---------------------------
def test_lf_coverage_with_strict_match_with_prefixes_without_agg(analysis_doc):
    """ Test expected coverage across below spans:
  
    Spans:
        "name_1": Pierre (B-PERSON), Lison (L-PERSON), Pierre(U-PERSON)
        "name_2": Pierre (B-PERSON), Lison (L-PERSON)
        "org_1": Norwegian (B-ORG), Computing (I-ORG), Center(L-ORG)
        "org_2": Norwegian (B-ORG), Computing (L-ORG), Oslo (U-ORG)
        "place_1": Norwegian (U-NORP), Oslo (U-GPE)

    Coverage:
        B-PERSON: 
            "name_1": 1/1
            "name_2": 1/1
            "org_1": 0/1
            "org_2": 0/1
            "place_1": 0/1
        L-PERSON: 
            "name_1": 1/1
            "name_2": 1/1
            "org_1": 0/1
            "org_2": 0/1
            "place_1": 0/1
        U-PERSON: 
            "name_1": 1/1
            "name_2": 0/1
            "org_1": 0/1
            "org_2": 0/1
            "place_1": 0/1
        B-ORG:
            "name_1": 0/1
            "name_2": 0/1
            "org_1": 1/1
            "org_2": 1/1
            "place_1": 0/1
        I-ORG:
            "name_1": 0/1
            "name_2": 0/1
            "org_1": 1/1
            "org_2": 0/1
            "place_1": 0/1
        L-ORG:
            "name_1": 0/2
            "name_2": 0/2
            "org_1": 1/2
            "org_2": 1/2
            "place_1": 0/1
        U-ORG:
            "name_1": 0/1
            "name_2": 0/1
            "org_1": 0/1
            "org_2": 1/1
            "place_1": 0/1
        U-NORP:
            "name_1": 0/1
            "name_2": 0/1
            "org_1": 0/1
            "org_2": 0/1
            "place_1": 1/1
        U-GPE:
            "name_1": 0/1
            "name_2": 0/1
            "org_1": 0/1
            "org_2": 0/1
            "place_1": 1/1
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=True
    )
    result = lf_analysis.lf_coverages(agg=False)

    assert result['name_1']['B-PERSON'] == 1.0
    assert result['name_2']['B-PERSON'] == 1.0
    assert result['org_1']['B-PERSON'] == 0.0
    assert result['org_2']['B-PERSON'] == 0.0
    assert result['place_1']['B-PERSON'] == 0.0
    assert result['name_1']['L-PERSON'] == 1.0
    assert result['name_2']['L-PERSON'] == 1.0
    assert result['org_1']['L-PERSON'] == 0.0
    assert result['org_2']['L-PERSON'] == 0.0
    assert result['place_1']['L-PERSON'] == 0.0 
    assert result['name_1']['U-PERSON'] == 1.0
    assert result['name_2']['U-PERSON'] == 0.0
    assert result['org_1']['U-PERSON'] == 0.0
    assert result['org_2']['U-PERSON'] == 0.0
    assert result['place_1']['U-PERSON'] == 0.0 

    assert result['name_1']['B-ORG'] == 0.0
    assert result['name_2']['B-ORG'] == 0.0
    assert result['org_1']['B-ORG'] == 1.0
    assert result['org_2']['B-ORG'] == 1.0
    assert result['place_1']['B-ORG'] == 0.0
    assert result['name_1']['I-ORG'] == 0.0
    assert result['name_2']['I-ORG'] == 0.0
    assert result['org_1']['I-ORG'] == 1.0
    assert result['org_2']['I-ORG'] == 0.0
    assert result['place_1']['I-ORG'] == 0.0
    assert result['name_1']['L-ORG'] == 0.0
    assert result['name_2']['L-ORG'] == 0.0
    assert result['org_1']['L-ORG'] == 1/2
    assert result['org_2']['L-ORG'] == 1/2
    assert result['place_1']['L-ORG'] == 0.0
    assert result['name_1']['U-ORG'] == 0.0
    assert result['name_2']['U-ORG'] == 0.0
    assert result['org_1']['U-ORG'] == 0.0
    assert result['org_2']['U-ORG'] == 1.0
    assert result['place_1']['U-ORG'] == 0.0

    assert result['name_1']['U-NORP'] == 0.0
    assert result['name_2']['U-NORP'] == 0.0
    assert result['org_1']['U-NORP'] == 0.0
    assert result['org_2']['U-NORP'] == 0.0
    assert result['place_1']['U-NORP'] == 1.0

    assert result['name_1']['U-GPE'] == 0.0
    assert result['name_2']['U-GPE'] == 0.0
    assert result['org_1']['U-GPE'] == 0.0
    assert result['org_2']['U-GPE'] == 0.0
    assert result['place_1']['U-GPE'] == 1.0


def test_lf_coverage_without_strict_match_with_prefixes_without_agg(
    analysis_doc
):
    """ Test expected coverage across below spans:
  
    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Coverage:
        PERSON: 
            "name_1": 3/3
            "name_2": 2/3
            "org_1": 0/3
            "org_2": 0/3
            "place_1": 0/3
        ORG:
            "name_1": 0/4
            "name_2": 0/4
            "org_1": 3/4
            "org_2": 3/4
            "place_1": 0/4
        NORP:
            "name_1": 0/1
            "name_2": 0/1
            "org_1": 0/1
            "org_2": 0/1
            "place_1": 1/1
        GPE:
            "name_1": 0/1
            "name_2": 0/1
            "org_1": 0/1
            "org_2": 0/1
            "place_1": 1/1
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_coverages(agg=False)

    assert result['name_1']['PERSON'] == 1.0
    assert result['name_1']['ORG'] == 0.0
    assert result['name_1']['NORP'] == 0.0
    assert result['name_1']['GPE'] == 0.0 

    assert result['name_2']['PERSON'] == 2/3
    assert result['name_2']['ORG'] == 0.0
    assert result['name_2']['NORP'] == 0.0
    assert result['name_2']['GPE'] == 0.0 

    assert result['org_1']['PERSON'] == 0.0
    assert result['org_1']['ORG'] == 3/4
    assert result['org_1']['NORP'] == 0.0
    assert result['org_1']['GPE'] == 0.0

    assert result['org_2']['PERSON'] == 0.0
    assert result['org_2']['ORG'] == 3/4
    assert result['org_2']['NORP'] == 0.0
    assert result['org_2']['GPE'] == 0.0

    assert result['place_1']['PERSON'] == 0.0
    assert result['place_1']['ORG'] == 0.0
    assert result['place_1']['NORP'] == 1.0
    assert result['place_1']['GPE'] == 1.0


def test_lf_coverage_without_strict_match_without_prefixes_without_agg(
    analysis_doc
):
    """ Test expected coverage across below spans:
  
    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Coverage:
        PERSON: 
            "name_1": 3/3
            "name_2": 2/3
            "org_1": 0/3
            "org_2": 0/3
            "place_1": 0/3
        ORG:
            "name_1": 0/4
            "name_2": 0/4
            "org_1": 3/4
            "org_2": 3/4
            "place_1": 0/4
        NORP:
            "name_1": 0/1
            "name_2": 0/1
            "org_1": 0/1
            "org_2": 0/1
            "place_1": 1/1
        GPE:
            "name_1": 0/1
            "name_2": 0/1
            "org_1": 0/1
            "org_2": 0/1
            "place_1": 1/1
    """
    labels = ["O", "GPE", "NORP", "ORG", "PERSON"]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_coverages(agg=False)

    assert result['name_1']['PERSON'] == 1.0
    assert result['name_1']['ORG'] == 0.0
    assert result['name_1']['NORP'] == 0.0
    assert result['name_1']['GPE'] == 0.0 

    assert result['name_2']['PERSON'] == 2/3
    assert result['name_2']['ORG'] == 0.0
    assert result['name_2']['NORP'] == 0.0
    assert result['name_2']['GPE'] == 0.0 

    assert result['org_1']['PERSON'] == 0.0
    assert result['org_1']['ORG'] == 3/4
    assert result['org_1']['NORP'] == 0.0
    assert result['org_1']['GPE'] == 0.0

    assert result['org_2']['PERSON'] == 0.0
    assert result['org_2']['ORG'] == 3/4
    assert result['org_2']['NORP'] == 0.0
    assert result['org_2']['GPE'] == 0.0

    assert result['place_1']['PERSON'] == 0.0
    assert result['place_1']['ORG'] == 0.0
    assert result['place_1']['NORP'] == 1.0
    assert result['place_1']['GPE'] == 1.0


# ------------------------
# LF COVERAGE TESTS w/ agg
# -------------------------
def test_lf_coverage_with_strict_match_with_prefixes_with_agg(analysis_doc):
    """ Test expected coverage across below spans:
  
    Spans:
        "name_1": Pierre (B-PERSON), Lison (L-PERSON), Pierre(U-PERSON)
        "name_2": Pierre (B-PERSON), Lison (L-PERSON)
        "org_1": Norwegian (B-ORG), Computing (I-ORG), Center(L-ORG)
        "org_2": Norwegian (B-ORG), Computing (L-ORG), Oslo (U-ORG)
        "place_1": Norwegian (U-NORP), Oslo (U-GPE)

    Coverage:
        "name_1": 3/3
        "name_2": 2/2
        "org_1": 3/3
        "org_2": 3/4
        "place_1": 2/2
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=True
    )
    result = lf_analysis.lf_coverages(agg=True)
    assert(result["name_1"].item() == 1.0)
    assert(result["name_2"].item() == 1.0)
    assert(result["org_1"].item() == 1.0)
    assert(result["org_2"].item() == 3/4)
    assert(result["place_1"].item() == 1.0)


def test_lf_coverage_without_strict_match_with_prefixes_with_agg(
    analysis_doc
):
    """ Test expected coverage across below spans:
  
    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Coverage:
        "name_1": 3/3
        "name_2": 2/3
        "org_1": 3/4
        "org_2": 3/4
        "place_1": 2/2
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_coverages(agg=True)
    assert(result["name_1"].item() == 1.0)
    assert(result["name_2"].item() == 2/3)
    assert(result["org_1"].item() == 3/4)
    assert(result["org_2"].item() == 3/4)
    assert(result["place_1"].item() == 1.0)


def test_lf_coverage_without_strict_match_without_prefixes_with_agg(
    analysis_doc
):
    """ Test expected coverage across below spans:
  
    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Coverage:
        "name_1": 3/3
        "name_2": 2/3
        "org_1": 3/4
        "org_2": 3/4
        "place_1": 2/2
    """
    labels = ["O", "GPE", "NORP", "ORG", "PERSON"]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_coverages(agg=True)
    assert(result["name_1"].item() == 1.0)
    assert(result["name_2"].item() == 2/3)
    assert(result["org_1"].item() == 3/4)
    assert(result["org_2"].item() == 3/4)
    assert(result["place_1"].item() == 1.0)


# ----------------------------
# LF OVERLAP TESTS w/out agg
# -----------------------------
def test_lf_overlaps_with_strict_match_with_prefixes_without_agg(analysis_doc):
    """ Test expected overlaps across below spans:
  
    Spans:
        "name_1": Pierre (B-PERSON), Lison (L-PERSON), Pierre(U-PERSON)
        "name_2": Pierre (B-PERSON), Lison (L-PERSON)
        "org_1": Norwegian (B-ORG), Computing (I-ORG), Center(L-ORG)
        "org_2": Norwegian (B-ORG), Computing (L-ORG), Oslo (U-ORG)
        "place_1": Norwegian (U-NORP), Oslo (U-GPE)

    Overlaps:
        name_1:
            B-PERSON: 1 Overlaps / 1
            L-PERSON: 1 Overlaps / 1
            U-PERSON: 0 Overlaps / 1
            B-ORG: 0 Overlaps / 0
            I-ORG: 0 Overlaps / 0
            L-ORG: 0 Overlaps / 0
            U-ORG: 0 Overlaps / 0
            U-NORP: 0 Overlaps / 0
            U-GPE: 0 Overlaps / 0
        name_2:
            B-PERSON: 1 Overlaps / 1
            L-PERSON: 1 Overlaps / 1
            U-PERSON: 0 Overlaps / 0
            B-ORG: 0 Overlaps / 0
            I-ORG: 0 Overlaps / 0
            L-ORG: 0 Overlaps / 0
            U-ORG: 0 Overlaps / 0
            U-NORP: 0 Overlaps / 0
            U-GPE: 0 Overlaps / 0
        org_1:
            B-PERSON: 0 Overlaps / 0
            L-PERSON: 0 Overlaps / 0
            U-PERSON: 0 Overlaps / 0
            B-ORG: 1 Overlap / 1
            I-ORG: 0 Overlap / 1 
            L-ORG: 0 Overlap / 1
            U-ORG: 0 Overlaps / 0
            U-NORP: 0 Overlaps / 0
            U-GPE: 0 Overlaps / 0
        org_2:
            B-PERSON: 0 Overlaps / 0
            L-PERSON: 0 Overlaps / 0
            U-PERSON: 0 Overlaps / 0
            B-ORG: 1 Overlap / 1
            I-ORG: 0 Overlaps / 0
            L-ORG: 1 Overlap / 1
            U-ORG: 1 Overlap / 1
            U-NORP: 0 Overlaps / 0
            U-GPE: 0 Overlaps / 0
        place_1:
            B-PERSON: 0 Overlaps / 0
            L-PERSON: 0 Overlaps / 0
            U-PERSON: 0 Overlaps / 0
            B-ORG: 1 Overlap / 1
            I-ORG: 0 Overlaps / 0
            L-ORG: 0 Conflict / 0
            U-ORG: 0 Conflict / 0
            U-NORP: 1 Overlap / 1
            U-GPE: 1 Overlap / 1
  
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=True
    )
    result = lf_analysis.lf_overlaps(agg=False)

    # Order:
    # U-GPE, U-NORP, B-ORG, I-ORG, L-ORG, U-ORG
    # B-PERSON, L-PERSON, U-PERSON
    assert (result['name_1'].to_list() == 
        [0., 0., 0., 0., 0., 0., 1., 1., 0.,])
    assert (result['name_2'].to_list() == 
        [0., 0., 0., 0., 0., 0., 1., 1., 0.,])
    assert (result['org_1'].to_list() == 
        [0., 0., 1., 1., 0., 0., 0., 0., 0.,])
    assert (result['org_2'].to_list() == 
        [0., 0., 1., 0., 1., 1., 0., 0., 0.,])
    assert (result['place_1'].to_list() == 
        [1., 1., 0., 0., 0., 0., 0., 0., 0.,])


def test_lf_overlaps_without_strict_match_with_prefixes_without_agg(
    analysis_doc
):
    """ Test expected overlaps across below spans:

    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Overlaps:
        name_1:
            PERSON: 2/3
            ORG: 0/0
            NORP: 0/0
            GPE: 0/0
        name_2:
            PERSON: 2/2
            ORG: 0/0
            NORP: 0/0
            GPE: 0/0
        org_1:
            PERSON: 0/0
            ORG: 2/3
            NORP: 0/0
            GPE: 0/0
        org_2:
            PERSON: 0/0
            ORG: 3/3
            NORP: 0/0
            GPE: 0/0
        place_1:
            PERSON: 0/0
            ORG: 0/0
            NORP: 1/1
            GPE: 1/1
  
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_overlaps(agg=False)
    # Order:
    # GPE, NORP, ORG, PERSON
    assert (result['name_1'].to_list() == 
        [0., 0., 0., 2/3])
    assert (result['name_2'].to_list() == 
        [0., 0., 0., 1.])
    assert (result['org_1'].to_list() == 
        [0., 0., 2/3, 0.])
    assert (result['org_2'].to_list() == 
        [0., 0., 1., 0.])
    assert (result['place_1'].to_list() == 
        [1., 1., 0., 0.])


def test_lf_overlaps_without_strict_match_without_prefixes_without_agg(
    analysis_doc
):
    """ Test expected overlaps across below spans:
  
    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Overlaps:
        name_1:
            PERSON: 2/3
            ORG: 0/0
            NORP: 0/0
            GPE: 0/0
        name_2:
            PERSON: 2/2
            ORG: 0/0
            NORP: 0/0
            GPE: 0/0
        org_1:
            PERSON: 0/0
            ORG: 2/3
            NORP: 0/0
            GPE: 0/0
        org_2:
            PERSON: 0/0
            ORG: 3/3
            NORP: 0/0
            GPE: 0/0
        place_1:
            PERSON: 0/0
            ORG: 0/0
            NORP: 1/1
            GPE: 1/1
  
    """
    labels = ["O", "GPE", "NORP", "ORG", "PERSON"]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_overlaps(agg=False)

    # Order:
    # GPE, NORP, ORG, PERSON
    assert (result['name_1'].to_list() == 
        [0., 0., 0., 2/3])
    assert (result['name_2'].to_list() == 
        [0., 0., 0., 1.])
    assert (result['org_1'].to_list() == 
        [0., 0., 2/3, 0.])
    assert (result['org_2'].to_list() == 
        [0., 0., 1., 0.])
    assert (result['place_1'].to_list() == 
        [1., 1., 0., 0.])


# -----------------------
# LF OVERLAP TESTS w/ agg
# -----------------------
def test_lf_overlaps_with_strict_match_with_prefixes_with_agg(analysis_doc):
    """ Test expected overlaps across below spans:
  
    Spans:
        "name_1": Pierre (B-PERSON), Lison (L-PERSON), Pierre(U-PERSON)
        "name_2": Pierre (B-PERSON), Lison (L-PERSON)
        "org_1": Norwegian (B-ORG), Computing (I-ORG), Center(L-ORG)
        "org_2": Norwegian (B-ORG), Computing (L-ORG), Oslo (U-ORG)
        "place_1": Norwegian (U-NORP), Oslo (U-GPE)

    Overlaps:
        "name_1": 2/3
        "name_2": 2/2
        "org_1": 2/3
        "org_2": 3/3
        "place_1": 2/2
  
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=True
    )
    result = lf_analysis.lf_overlaps(agg=True)
    assert result['name_1'].item() == 2/3
    assert result['name_2'].item() == 1.0
    assert result['org_1'].item() == 2/3
    assert result['org_2'].item() == 1.0
    assert result['place_1'].item() == 1.0


def test_lf_overlaps_without_strict_match_with_prefixes_with_agg(analysis_doc):
    """ Test expected overlaps across below spans:

    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Overlaps:
        "name_1": 2/3
        "name_2": 2/2
        "org_1": 2/3
        "org_2": 3/3
        "place_1": 2/2
  
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_overlaps(agg=True)
    assert result['name_1'].item() == 2/3
    assert result['name_2'].item() == 1.0
    assert result['org_1'].item() == 2/3
    assert result['org_2'].item() == 1.0
    assert result['place_1'].item() == 1.0


def test_lf_overlaps_without_strict_match_without_prefixes_with_agg(
    analysis_doc
):
    """ Test expected overlaps across below spans:
  
    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Overlaps:
        "name_1": 2/3
        "name_2": 2/2
        "org_1": 2/3
        "org_2": 3/3
        "place_1": 2/2
  
    """
    labels = ["O", "GPE", "NORP", "ORG", "PERSON"]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_overlaps(agg=True)
    assert result['name_1'].item() == 2/3
    assert result['name_2'].item() == 1.0
    assert result['org_1'].item() == 2/3
    assert result['org_2'].item() == 1.0
    assert result['place_1'].item() == 1.0


# ---------------------------
# LF CONFLICT TESTS w/out Agg
# ---------------------------
def test_lf_conflicts_with_strict_match_with_prefixes_without_agg(
    analysis_doc
):
    """ Test expected conflicts across below spans:
  
    Spans:
        "name_1": Pierre (B-PERSON), Lison (L-PERSON), Pierre(U-PERSON)
        "name_2": Pierre (B-PERSON), Lison (L-PERSON)
        "org_1": Norwegian (B-ORG), Computing (I-ORG), Center(L-ORG)
        "org_2": Norwegian (B-ORG), Computing (L-ORG), Oslo (U-ORG)
        "place_1": Norwegian (U-NORP), Oslo (U-GPE)

    Conflicts:
        name_1:
            B-PERSON: 0 Conflicts / 1
            L-PERSON: 0 Conflicts / 1
            U-PERSON: 0 Conflicts / 1
            B-ORG: 0 Conflicts / 0
            I-ORG: 0 Conflicts / 0
            L-ORG: 0 Conflicts / 0
            U-ORG: 0 Conflicts / 0
            U-NORP: 0 Conflicts / 0
            U-GPE: 0 Conflicts / 0
        name_2:
            B-PERSON: 0 Conflicts / 1
            L-PERSON: 0 Conflicts / 1
            U-PERSON: 0 Conflicts / 0
            B-ORG: 0 Conflicts / 0
            I-ORG: 0 Conflicts / 0
            L-ORG: 0 Conflicts / 0
            U-ORG: 0 Conflicts / 0
            U-NORP: 0 Conflicts / 0
            U-GPE: 0 Conflicts / 0
        org_1:
            B-PERSON: 0 Conflicts / 0
            L-PERSON: 0 Conflicts / 0
            U-PERSON: 0 Conflicts / 0
            B-ORG: 1 Conflict / 1
            I-ORG: 1 Conflict / 1 
            L-ORG: 0 Conflict / 1
            U-ORG: 0 Conflicts / 0
            U-NORP: 0 Conflicts / 0
            U-GPE: 0 Conflicts / 0
        org_2:
            B-PERSON: 0 Conflicts / 0
            L-PERSON: 0 Conflicts / 0
            U-PERSON: 0 Conflicts / 0
            B-ORG: 1 Conflict / 1
            I-ORG: 0 Conflicts / 0
            L-ORG: 1 Conflict / 1
            U-ORG: 1 Conflict / 1
            U-NORP: 0 Conflicts / 0
            U-GPE: 0 Conflicts / 0
        place_1:
            B-PERSON: 0 Conflicts / 0
            L-PERSON: 0 Conflicts / 0
            U-PERSON: 0 Conflicts / 0
            B-ORG: 1 Conflict / 1
            I-ORG: 0 Conflicts / 0
            L-ORG: 0 Conflict / 0
            U-ORG: 0 Conflict / 0
            U-NORP: 1 Conflict / 1
            U-GPE: 1 Conflict / 1
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=True
    )
    result = lf_analysis.lf_conflicts(agg=False)
    assert (result['name_1'].to_list() == 
        [0., 0., 0., 0., 0., 0., 0., 0., 0.,])
    assert (result['name_2'].to_list() == 
        [0., 0., 0., 0., 0., 0., 0., 0., 0.,])
    assert (result['org_1'].to_list() == 
        [0., 0., 1., 1., 0., 0., 0., 0., 0.,])
    assert (result['org_2'].to_list() == 
        [0., 0., 1., 0., 1., 1., 0., 0., 0.,])
    assert (result['place_1'].to_list() == 
        [1., 1., 0., 0., 0., 0., 0., 0., 0.,])


def test_lf_conflicts_without_strict_match_with_prefixes_without_agg(
    analysis_doc
):
    """ Test expected conflicts across below spans:

    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Conflicts:
        name_1:
            PERSON: 0/3
            ORG: 0/0
            NORP: 0/0
            GPE: 0/0
        name_2:
            PERSON: 0/2
            ORG: 0/0
            NORP: 0/0
            GPE: 0/0
        org_1:
            PERSON: 0/0
            ORG: 1/3
            NORP: 0/0
            GPE: 0/0
        org_2:
            PERSON: 0/0
            ORG: 2/3
            NORP: 0/0
            GPE: 0/0
        place_1:
            PERSON: 0/0
            ORG: 0/0
            NORP: 1/1
            GPE: 1/1
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_conflicts(agg=False)
    assert (result['name_1'].to_list() == 
        [0., 0., 0., 0.,])
    assert (result['name_2'].to_list() == 
        [0., 0., 0., 0.,])
    assert (result['org_1'].to_list() == 
        [0., 0., 1/3, 0.,])
    assert (result['org_2'].to_list() == 
        [0., 0., 2/3., 0.,])
    assert (result['place_1'].to_list() == 
        [1., 1., 0., 0.,])
    

def test_lf_conflicts_without_strict_match_without_prefixes_without_agg(
    analysis_doc
):
    """ Test expected conflicts across below spans:

    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Conflicts:
        name_1:
            PERSON: 0/3
            ORG: 0/0
            NORP: 0/0
            GPE: 0/0
        name_2:
            PERSON: 0/2
            ORG: 0/2
            NORP: 0/0
            GPE: 0/0
        org_1:
            PERSON: 0/0
            ORG: 1/3
            NORP: 0/0
            GPE: 0/0
        org_2:
            PERSON: 0/0
            ORG: 2/3
            NORP: 0/0
            GPE: 0/0
        place_1:
            PERSON: 0/0
            ORG: 0/0
            NORP: 1/1
            GPE: 1/1
    """
    labels = ["O", "GPE", "NORP", "ORG", "PERSON"]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_conflicts(agg=False)
    assert (result['name_1'].to_list() == 
        [0., 0., 0., 0.,])
    assert (result['name_2'].to_list() == 
        [0., 0., 0., 0.,])
    assert (result['org_1'].to_list() == 
        [0., 0., 1/3, 0.,])
    assert (result['org_2'].to_list() == 
        [0., 0., 2/3., 0.,])
    assert (result['place_1'].to_list() == 
        [1., 1., 0., 0.,])


# ------------------------
# LF CONFLICT TESTS w/ Agg
# ------------------------
def test_lf_conflicts_with_strict_match_with_prefixes_with_agg(
    analysis_doc
):
    """ Test expected conflicts across below spans:

    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

        name_1: 0/3
        name_2: 0/2
        org_1: 2/3
        org_2: 3/3
        place_1: 3/3
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=True
    )
    result = lf_analysis.lf_conflicts(agg=True)
    assert (result['name_1'].item() == 0.0)
    assert (result['name_2'].item() == 0.0)
    assert (result['org_1'].item() == 2/3)
    assert (result['org_2'].item() == 1.0)
    assert (result['place_1'].item() == 1.0)


def test_lf_conflicts_without_strict_match_with_prefixes_with_agg(
    analysis_doc
):
    """ Test expected conflicts across below spans:

    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Conflicts:
        name_1: 0/3
        name_2: 0/2
        org_1: 1/3
        org_2: 2/3
        place_1: 2/2
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_conflicts(agg=True)
    assert (result['name_1'].item() == 0.0)
    assert (result['name_2'].item() == 0.0)
    assert (result['org_1'].item() == 1/3)
    assert (result['org_2'].item() == 2/3)
    assert (result['place_1'].item() == 1.0)


def test_lf_conflicts_without_strict_match_without_prefixes_with_agg(
    analysis_doc
):
    """ Test expected conflicts across below spans:

    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Conflicts:
        name_1: 0/3
        name_2: 0/2
        org_1: 1/3
        org_2: 2/3
        place_1: 2/2
    """
    labels = ["O", "GPE", "NORP", "ORG", "PERSON"]
    lf_analysis = LFAnalysis(
        [analysis_doc],
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_conflicts(agg=True)
    assert (result['name_1'].item() == 0.0)
    assert (result['name_2'].item() == 0.0)
    assert (result['org_1'].item() == 1/3)
    assert (result['org_2'].item() == 2/3)
    assert (result['place_1'].item() == 1.0)
