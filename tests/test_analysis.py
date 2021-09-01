import re

import numpy as np
from spacy.tokens import Span #type: ignore
import pytest

from skweak.analysis import LFAnalysis


@pytest.fixture(scope="session")
def analysis_corpus(nlp):
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
    return [spacy_doc]


@pytest.fixture(scope="session")
def analysis_corpus_y(nlp):
    """ Generate ground truth for the `analysis_corpus`
    """
    spacy_doc = nlp(
        re.sub("\\s+", " ", """This is a test for Pierre Lison from the
                     Norwegian Computing Center. Pierre is living in Oslo."""))
    spacy_doc.spans["ground_truth"] = [
        Span(spacy_doc, 5, 7, label="PERSON"),
        Span(spacy_doc, 13, 14, label="PERSON"),
        Span(spacy_doc, 9, 12, label="ORG"),
        Span(spacy_doc, 17, 18, label="GPE")
    ]
    return [spacy_doc], "ground_truth", ["O", "PERSON", "ORG", "GPE"]


# ---------------------
# LABEL OVERLAP TESTS
# ---------------------
def test_overlaps_with_strict_match_with_prefixes(analysis_corpus):
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
        analysis_corpus,
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


def test_overlaps_without_strict_match_with_prefixes(analysis_corpus):
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
        analysis_corpus,
        labels,
        strict_match=False
    )
    result = lf_analysis.label_overlap()
    assert result['overlap']['PERSON'] == 2/3
    assert result['overlap']['ORG'] == 3/4
    assert result['overlap']['NORP'] == 1.0
    assert result['overlap']['GPE'] == 1.0


def test_overlaps_without_strict_match_without_prefixes(analysis_corpus):
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
        analysis_corpus,
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
def test_conflicts_with_strict_match_with_prefixes(analysis_corpus):
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
        analysis_corpus,
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


def test_conflicts_without_strict_match_with_prefixes(analysis_corpus):
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
        analysis_corpus,
        labels,
        strict_match=False
    )
    result = lf_analysis.label_conflict()
    assert result['conflict']['PERSON'] == 0.0
    assert result['conflict']['ORG'] == 2/4
    assert result['conflict']['NORP'] == 1.0
    assert result['conflict']['GPE'] == 1.0


def test_conflicts_without_strict_match_without_prefixes(analysis_corpus):
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
        analysis_corpus,
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
def test_lf_targets_with_strict_match_with_prefixes(analysis_corpus):
    """ Test expected targets across below spans:

    Spans:
        "name_1": Pierre (B-PERSON), Lison (L-PERSON), Pierre(U-PERSON)
        "name_2": Pierre (B-PERSON), Lison (L-PERSON)
        "org_1": Norwegian (B-ORG), Computing (I-ORG), Center(L-ORG)
        "org_2": Norwegian (B-ORG), Computing (L-ORG), Oslo (U-ORG)
        "place_1": Norwegian (U-NORP), Oslo (U-GPE)

    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        analysis_corpus,
        labels,
        strict_match=True
    )
    result = lf_analysis.lf_target_labels()

    assert (
        set(result["name_1"]) == 
        set([
            lf_analysis.label2idx["B-PERSON"],
            lf_analysis.label2idx["I-PERSON"],
            lf_analysis.label2idx["L-PERSON"],
            lf_analysis.label2idx["U-PERSON"],
        ])
    )
    assert (
        set(result["name_2"]) == 
        set([
            lf_analysis.label2idx["B-PERSON"],
            lf_analysis.label2idx["I-PERSON"],
            lf_analysis.label2idx["L-PERSON"],
            lf_analysis.label2idx["U-PERSON"],
        ])
    )
    assert (
        set(result["org_1"]) == 
        set([
            lf_analysis.label2idx["B-ORG"],
            lf_analysis.label2idx["I-ORG"],
            lf_analysis.label2idx["L-ORG"],
            lf_analysis.label2idx["U-ORG"],
        ])
    )
    assert (
        set(result["org_2"]) == 
        set([
            lf_analysis.label2idx["B-ORG"],
            lf_analysis.label2idx["I-ORG"],
            lf_analysis.label2idx["L-ORG"],
            lf_analysis.label2idx["U-ORG"],
        ])
    )
    assert (
        set(result["place_1"]) == 
        set([
            lf_analysis.label2idx["B-NORP"],
            lf_analysis.label2idx["I-NORP"],
            lf_analysis.label2idx["L-NORP"],
            lf_analysis.label2idx["U-NORP"],
            lf_analysis.label2idx["B-GPE"],
            lf_analysis.label2idx["I-GPE"],
            lf_analysis.label2idx["L-GPE"],
            lf_analysis.label2idx["U-GPE"],
        ])
    ) 


def test_lf_targets_without_strict_match_with_prefixes(analysis_corpus):
    """ Test expected targets across below spans:

    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        analysis_corpus,
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


def test_lf_targets_without_strict_match_without_prefixes(analysis_corpus):
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
        analysis_corpus,
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
def test_lf_coverage_with_strict_match_with_prefixes_without_agg(analysis_corpus):
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
        analysis_corpus,
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
    analysis_corpus
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
        analysis_corpus,
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
    analysis_corpus
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
        analysis_corpus,
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
def test_lf_coverage_with_strict_match_with_prefixes_with_agg(analysis_corpus):
    """ Test expected coverage across below spans:
  
    Spans:
        "name_1": Pierre (B-PERSON), Lison (L-PERSON), Pierre(U-PERSON)
        "name_2": Pierre (B-PERSON), Lison (L-PERSON)
        "org_1": Norwegian (B-ORG), Computing (I-ORG), Center(L-ORG)
        "org_2": Norwegian (B-ORG), Computing (L-ORG), Oslo (U-ORG)
        "place_1": Norwegian (U-NORP), Oslo (U-GPE)

    Coverage:
        "name_1": 3/3 (Target Set: (BILU)-PERSON)
        "name_2": 2/3 (Target Set: (BILU)-PERSON)
        "org_1": 3/4 (Target Set: (BILU)-ORG)
        "org_2": 3/4 (Target Set: (BILU)-ORG)
        "place_1": 2/2 (Target Set: (BILU)-NORP, (BILU)-GPE)
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        analysis_corpus,
        labels,
        strict_match=True
    )
    result = lf_analysis.lf_coverages(agg=True)
    assert(result["name_1"].item() == 1.0)
    assert(result["name_2"].item() == 2/3)
    assert(result["org_1"].item() == 3/4)
    assert(result["org_2"].item() == 3/4)
    assert(result["place_1"].item() == 1.0)


def test_lf_coverage_without_strict_match_with_prefixes_with_agg(
    analysis_corpus
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
        analysis_corpus,
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
    analysis_corpus
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
        analysis_corpus,
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
def test_lf_overlaps_with_strict_match_with_prefixes_without_agg(
    analysis_corpus
):
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
            I-ORG: 1 Overlap / 1 
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
            B-ORG: 0 Overlap / 0
            I-ORG: 0 Overlaps / 0
            L-ORG: 0 Overlap / 0
            U-ORG: 0 Overlap / 0
            U-NORP: 1 Overlap / 1
            U-GPE: 1 Overlap / 1
  
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        analysis_corpus,
        labels,
        strict_match=True
    )
    result = lf_analysis.lf_overlaps(agg=False, nan_to_num=-1.0)
    
    # Create expected arrays
    default = np.ones(16) * -1
    name_1_exp = default.copy()
    name_2_exp = default.copy()
    org_1_exp = default.copy()
    org_2_exp = default.copy()
    place_1_exp = default.copy()

    # Order:
    # B-GPE, I-GPE, L-GPE, U-GPE,
    # B-NORP, I-NORP, L-NORP, U-NORP,
    # B-ORG, I-ORG, L-ORG, U-ORG,
    # B-PERSON, I-PERSON, L-PERSON, U-PERSON

    name_1_exp[12:16] = [1., -1., 1., 0.,]
    name_2_exp[12:15] = [1., -1., 1.,]
    org_1_exp[8:12] = [1., 1., 0., -1]
    org_2_exp[8:12] = [1., -1., 1., 1.,]
    place_1_exp[3] = 1.
    place_1_exp[7] = 1.

    # Check
    np.testing.assert_allclose(result['name_1'].to_numpy(), name_1_exp)
    np.testing.assert_allclose(result['name_2'].to_numpy(), name_2_exp)
    np.testing.assert_allclose(result['org_1'].to_numpy(), org_1_exp)
    np.testing.assert_allclose(result['org_2'].to_numpy(), org_2_exp)
    np.testing.assert_allclose(result['place_1'].to_numpy(), place_1_exp)


def test_lf_overlaps_without_strict_match_with_prefixes_without_agg(
    analysis_corpus
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
        analysis_corpus,
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
    analysis_corpus
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
        analysis_corpus,
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
def test_lf_overlaps_with_strict_match_with_prefixes_with_agg(analysis_corpus):
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
        analysis_corpus,
        labels,
        strict_match=True
    )
    result = lf_analysis.lf_overlaps(agg=True)
    assert result['name_1'].item() == 2/3
    assert result['name_2'].item() == 1.0
    assert result['org_1'].item() == 2/3
    assert result['org_2'].item() == 1.0
    assert result['place_1'].item() == 1.0


def test_lf_overlaps_without_strict_match_with_prefixes_with_agg(analysis_corpus):
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
        analysis_corpus,
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
    analysis_corpus
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
        analysis_corpus,
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
    analysis_corpus
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
            B-ORG: 0 Conflict / 0
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
        analysis_corpus,
        labels,
        strict_match=True
    )
    result = lf_analysis.lf_conflicts(agg=False, nan_to_num=-1.)

   # Create expected arrays
    default = np.ones(16) * -1
    name_1_exp = default.copy()
    name_2_exp = default.copy()
    org_1_exp = default.copy()
    org_2_exp = default.copy()
    place_1_exp = default.copy()

    # Order:
    # B-GPE, I-GPE, L-GPE, U-GPE,
    # B-NORP, I-NORP, L-NORP, U-NORP,
    # B-ORG, I-ORG, L-ORG, U-ORG,
    # B-PERSON, I-PERSON, L-PERSON, U-PERSON

    name_1_exp[12:16] = [0., -1., 0., 0.,]
    name_2_exp[12:15] = [0., -1., 0.,]
    org_1_exp[8:12] = [1., 1., 0., -1]
    org_2_exp[8:12] = [1., -1., 1., 1.,]
    place_1_exp[3] = 1.
    place_1_exp[7] = 1.

    # Check
    np.testing.assert_allclose(result['name_1'].to_numpy(), name_1_exp)
    np.testing.assert_allclose(result['name_2'].to_numpy(), name_2_exp)
    np.testing.assert_allclose(result['org_1'].to_numpy(), org_1_exp)
    np.testing.assert_allclose(result['org_2'].to_numpy(), org_2_exp)
    np.testing.assert_allclose(result['place_1'].to_numpy(), place_1_exp)


def test_lf_conflicts_without_strict_match_with_prefixes_without_agg(
    analysis_corpus
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
        analysis_corpus,
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
    analysis_corpus
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
        analysis_corpus,
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
    analysis_corpus
):
    """ Test expected conflicts across below spans:

    Spans:
        "name_1": Pierre (B-PERSON), Lison (L-PERSON), Pierre(U-PERSON)
        "name_2": Pierre (B-PERSON), Lison (L-PERSON)
        "org_1": Norwegian (B-ORG), Computing (I-ORG), Center(L-ORG)
        "org_2": Norwegian (B-ORG), Computing (L-ORG), Oslo (U-ORG)
        "place_1": Norwegian (U-NORP), Oslo (U-GPE)

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
        analysis_corpus,
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
    analysis_corpus
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
        analysis_corpus,
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
    analysis_corpus
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
        analysis_corpus,
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_conflicts(agg=True)
    assert (result['name_1'].item() == 0.0)
    assert (result['name_2'].item() == 0.0)
    assert (result['org_1'].item() == 1/3)
    assert (result['org_2'].item() == 2/3)
    assert (result['place_1'].item() == 1.0)

# ------------------------
# LF ACCURACY TESTS w/ Agg
# ------------------------
def test_lf_accs_with_strict_match_with_prefixes_with_agg(
    analysis_corpus,
    analysis_corpus_y,
):
    """ Test expected accuracies across below spans:

    Spans:
        "name_1": Pierre (B-PERSON), Lison (L-PERSON), Pierre(U-PERSON)
        "name_2": Pierre (B-PERSON), Lison (L-PERSON)
        "org_1": Norwegian (B-ORG), Computing (I-ORG), Center(L-ORG)
        "org_2": Norwegian (B-ORG), Computing (L-ORG), Oslo (U-ORG)
        "place_1": Norwegian (U-NORP), Oslo (U-GPE)

    Accuracy:
        name_1: 19/19 
        name_2: 18/19 (misclassifies second Pierre)
        org_1: 19/19 
        org_2: 16/19 (misclassifies Computing, Center, and Oslo)
        place_1: 18/19 (misclassifies Norwegian)
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        analysis_corpus,
        labels,
        strict_match=True
    )
    result = lf_analysis.lf_empirical_accuracies(
        *analysis_corpus_y,
        agg=True
    )
    assert(result['acc']['name_1'] == 1.0)
    assert(abs(result['acc']['name_2'] - 18/19) <= 1e-5)
    assert(result['acc']['org_1'] == 1.0)
    assert(abs(result['acc']['org_2'] - 16/19) <= 1e-5)
    assert(abs(result['acc']['place_1'] - 18/19) <= 1e-5)


def test_lf_acc_without_strict_match_with_prefixes_with_agg(
    analysis_corpus,
    analysis_corpus_y
):
    """ Test expected accuracies across below spans:

    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Accuracy:
        name_1: 19/19 (gets all PERSON tags)
        name_2: 18/19 (misclassifies second Pierre)
        org_1: 19/19 (gets all ORG tags)
        org_2: 17/19 (misclassifies Center and Oslo)
        place_1: 18/19
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        analysis_corpus,
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_empirical_accuracies(
        *analysis_corpus_y,
        agg=True
    )
    assert(result['acc']['name_1'] == 1.0)
    assert(abs(result['acc']['name_2'] - 18/19) <= 1e-5)
    assert(result['acc']['org_1'] == 1.0)
    assert(abs(result['acc']['org_2'] - 17/19) <= 1e-5)
    assert(abs(result['acc']['place_1'] - 18/19) <= 1e-5)


def test_lf_acc_without_strict_match_without_prefixes_with_agg(
    analysis_corpus,
    analysis_corpus_y
):
    """ Test expected accuracies across below spans:

    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Accuracy:
        name_1: 19/19 (gets all PERSON tags)
        name_2: 18/19 (misclassifies second Pierre)
        org_1: 19/19 (gets all ORG tags)
        org_2: 17/19 (misclassifies Center and Oslo)
        place_1: 18/19
    """
    labels = ["O", "GPE", "NORP", "ORG", "PERSON"]
    lf_analysis = LFAnalysis(
        analysis_corpus,
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_empirical_accuracies(
        *analysis_corpus_y,
        agg=True
    )
    assert(result['acc']['name_1'] == 1.0)
    assert(abs(result['acc']['name_2'] - 18/19) <= 1e-5)
    assert(result['acc']['org_1'] == 1.0)
    assert(abs(result['acc']['org_2'] - 17/19) <= 1e-5)
    assert(abs(result['acc']['place_1'] - 18/19) <= 1e-5)


# ----------------------------
# LF ACCURACY TESTS w/out Agg
# ----------------------------
def test_lf_acc_with_strict_match_with_prefixes_without_agg(
    analysis_corpus,
    analysis_corpus_y
):
    """ Test expected accuracies across below spans:

    Spans:
        "name_1": Pierre (B-PERSON), Lison (L-PERSON), Pierre(U-PERSON)
        "name_2": Pierre (B-PERSON), Lison (L-PERSON)
        "org_1": Norwegian (B-ORG), Computing (I-ORG), Center(L-ORG)
        "org_2": Norwegian (B-ORG), Computing (L-ORG), Oslo (U-ORG)
        "place_1": Norwegian (U-NORP), Oslo (U-GPE)

    Accuracy:
        name_1:
            B-PERSON: 19/19
            I-PERSON: 19/19
            L-PERSON: 19/19
            U-PERSON: 19/19
        name_2:
            B-PERSON: 19/19
            I-PERSON: 19/19
            L-PERSON: 19/19
            U-PERSON: 18/19 (misclassifies 2nd Pierre)
        org_1:
            B-ORG: 19/19
            I-ORG: 19/19
            L-ORG: 19/19
            U-ORG: 19/19 
        org_2:
            B-ORG: 19/19
            I-ORG: 18/19
            L-ORG: 17/19
            U-ORG: 18/19 
        place_1:
            B-NORP: 19/19
            I-NORP: 19/19
            L-NORP: 19/19
            U-NORP: 18/19 
            B-GPE: 19/19
            I-GPE: 19/19
            L-GPE: 19/19
            U-GPE: 19/19 
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        analysis_corpus,
        labels,
        strict_match=True
    )
    result = lf_analysis.lf_empirical_accuracies(
        *analysis_corpus_y,
        agg=False
    )
    assert(result['B-PERSON']['name_1'] == 1.0)
    assert(result['I-PERSON']['name_1'] == 1.0)
    assert(result['L-PERSON']['name_1'] == 1.0)
    assert(result['U-PERSON']['name_1'] == 1.0)

    assert(result['B-PERSON']['name_2'] == 1.0)
    assert(result['I-PERSON']['name_2'] == 1.0)
    assert(result['L-PERSON']['name_2'] == 1.0)
    assert(abs(result['U-PERSON']['name_2'] - 18/19) <= 1e-5)

    assert(result['B-ORG']['org_1'] == 1.0)
    assert(result['I-ORG']['org_1'] == 1.0)
    assert(result['L-ORG']['org_1'] == 1.0)
    assert(result['U-ORG']['org_1'] == 1.0)

    assert(result['B-ORG']['org_2'] == 1.0)
    assert(abs(result['I-ORG']['org_2'] - 18/19) <= 1e-5)
    assert(abs(result['L-ORG']['org_2'] - 17/19) <= 1e-5)
    assert(abs(result['U-ORG']['org_2'] - 18/19) <= 1e-5)

    assert(result['B-NORP']['place_1'] == 1.0)
    assert(result['I-NORP']['place_1'] == 1.0)
    assert(result['L-NORP']['place_1'] == 1.0)
    assert(abs(result['U-NORP']['place_1'] - 18/19) <= 1e-5)

    assert(result['B-GPE']['place_1'] == 1.0)
    assert(result['I-GPE']['place_1'] == 1.0)
    assert(result['L-GPE']['place_1'] == 1.0)
    assert(result['U-GPE']['place_1'] == 1.0)


def test_lf_acc_without_strict_match_with_prefixes_without_agg(
    analysis_corpus,
    analysis_corpus_y
):
    """ Test expected accuracies across below spans:

    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Accuracy:
        name_1:
            PERSON: 19/19
        name_2:
            PERSON: 18/19
        org_1:
            ORG: 19/19
        org_2:
            ORG: 17/19
        place_1:
            NORP: 18/19
            GPE: 19/19
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        analysis_corpus,
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_empirical_accuracies(
        *analysis_corpus_y,
        agg=False
    )
    assert(result['GPE']['place_1'] == 1.0)
    assert(abs(result['NORP']['place_1'] - 18/19) <= 1e-5)
    assert(abs(result['ORG']['org_2'] - 17/19) <= 1e-5)
    assert(result['ORG']['org_1'] == 1.0)
    assert(result['PERSON']['name_1'] == 1.0)
    assert(abs(result['PERSON']['name_2'] - 18/19) <= 1e-5)



def test_lf_acc_without_strict_match_without_prefixes_without_agg(
    analysis_corpus,
    analysis_corpus_y
):
    """ Test expected accuracies across below spans:

    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Accuracy:
        name_1:
            PERSON: 19/19
        name_2:
            PERSON: 18/19
        org_1:
            ORG: 19/19
        org_2:
            ORG: 17/19
        place_1:
            NORP: 18/19
            GPE: 19/19
    """
    labels = ["O", "GPE", "NORP", "ORG", "PERSON"]
    lf_analysis = LFAnalysis(
        analysis_corpus,
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_empirical_accuracies(
        *analysis_corpus_y,
        agg=False
    )
    assert(result['GPE']['place_1'] == 1.0)
    assert(abs(result['NORP']['place_1'] - 18/19) <= 1e-5)
    assert(abs(result['ORG']['org_2'] - 17/19) <= 1e-5)
    assert(result['ORG']['org_1'] == 1.0)
    assert(result['PERSON']['name_1'] == 1.0)
    assert(abs(result['PERSON']['name_2'] - 18/19) <= 1e-5)


# ----------------------------
# LF P, R, F1 TESTS w/ Agg
# ----------------------------
def test_lf_scores_with_strict_match_with_prefixes_with_agg(
    analysis_corpus,
    analysis_corpus_y
):
    """ Test expected conflicts across below spans:

    Spans:
        "name_1": Pierre (B-PERSON), Lison (L-PERSON), Pierre(U-PERSON)
        "name_2": Pierre (B-PERSON), Lison (L-PERSON)
        "org_1": Norwegian (B-ORG), Computing (I-ORG), Center(L-ORG)
        "org_2": Norwegian (B-ORG), Computing (L-ORG), Oslo (U-ORG)
        "place_1": Norwegian (U-NORP), Oslo (U-GPE)

    Precision, Recall:
        name_1: (P, R) - 3/3, 3/3
        name_2: (P, R) - 2/2, 2/3
        org_1: (P, R) - 3/3, 3/3
        org_2: (P, R) - 1/3, 1/3
        place_1: (P, R) - 1/1, 1/1
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        analysis_corpus,
        labels,
        strict_match=True
    )
    result = lf_analysis.lf_empirical_scores(
        *analysis_corpus_y,
        agg=True,
        nan_to_num=-1
    )

    assert(result['name_1']['precision'] == 1.)
    assert(result['name_1']['recall'] == 1.)
    assert(result['name_2']['precision'] == 1.)
    assert(result['name_2']['recall'] == 2/3)
    assert(result['org_1']['precision'] == 1.)
    assert(result['org_1']['recall'] == 1.)
    assert(result['org_2']['precision'] == 1/3)
    assert(result['org_2']['recall'] == 1/3)
    assert(result['place_1']['precision'] == 1.)
    assert(result['place_1']['recall'] == 1.)
    assert('NORP' not in result['place_1'].keys())


def test_lf_scores_without_strict_match_with_prefixes_with_agg(
    analysis_corpus,
    analysis_corpus_y
):
    """ Test expected precision and recall scores across below spans:

    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Precision, Recall :
        name_1: (P, R) - 3/3, 3/3
        name_2: (P, R) - 2/2, 2/3
        org_1: (P, R) - 3/3, 3/3
        org_2: (P, R) - 2/3, 2/3
        place_1: (P, R) - 1/1, 1/1
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        analysis_corpus,
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_empirical_scores(
        *analysis_corpus_y,
        agg=True
    )
    assert(result['name_1']['precision'] == 1.)
    assert(result['name_1']['recall'] == 1.)
    assert(result['name_2']['precision'] == 1.)
    assert(result['name_2']['recall'] == 2/3)
    assert(result['org_1']['precision'] == 1.)
    assert(result['org_1']['recall'] == 1.)
    assert(result['org_2']['precision'] == 2/3)
    assert(result['org_2']['recall'] == 2/3)
    assert(result['place_1']['precision'] == 1.)
    assert(result['place_1']['recall'] == 1.)
    assert('NORP' not in result['place_1'].keys())


def test_lf_scores_without_strict_match_without_prefixes_with_agg(
    analysis_corpus,
    analysis_corpus_y
):
    """ Test expected precision and recall across below spans:

    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Precision, Recall:
        name_1: (P, R) - 3/3, 3/3
        name_2: (P, R) - 2/2, 2/3
        org_1: (P, R) - 3/3, 3/3
        org_2: (P, R) - 2/3, 2/3
        place_1: (P, R) - 1/1, 1/1
    """
    labels = ["O", "GPE", "NORP", "ORG", "PERSON"]
    lf_analysis = LFAnalysis(
        analysis_corpus,
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_empirical_scores(
        *analysis_corpus_y,
        agg=True
    )
    assert(result['name_1']['precision'] == 1.)
    assert(result['name_1']['recall'] == 1.)
    assert(result['name_2']['precision'] == 1.)
    assert(result['name_2']['recall'] == 2/3)
    assert(result['org_1']['precision'] == 1.)
    assert(result['org_1']['recall'] == 1.)
    assert(result['org_2']['precision'] == 2/3)
    assert(result['org_2']['recall'] == 2/3)
    assert(result['place_1']['precision'] == 1.)
    assert(result['place_1']['recall'] == 1.)
    assert('NORP' not in result['place_1'].keys())


# ----------------------------
# LF P, R, F1 TESTS w/out Agg
# ----------------------------
def test_lf_scores_with_strict_match_with_prefixes_without_agg(
    analysis_corpus,
    analysis_corpus_y
):
    """ Test expected conflicts across below spans:

    Spans:
        "name_1": Pierre (B-PERSON), Lison (L-PERSON), Pierre(U-PERSON)
        "name_2": Pierre (B-PERSON), Lison (L-PERSON)
        "org_1": Norwegian (B-ORG), Computing (I-ORG), Center(L-ORG)
        "org_2": Norwegian (B-ORG), Computing (L-ORG), Oslo (U-ORG)
        "place_1": Norwegian (U-NORP), Oslo (U-GPE)

    Precision, Recall:
        name_1:
            B-PERSON (P, R): 1/1, 1/1
            I-PERSON (P, R): 0/0, 0/0
            L-PERSON (P, R): 1/1, 1/1
            U-PERSON (P, R): 1/1, 1/1
        name_2:
            B-PERSON (P, R): 1/1, 1/1
            I-PERSON (P, R): 0/0 (nan), 0/0 (nan)
            L-PERSON (P, R): 1/1, 1/1
            U-PERSON (P, R): 0/0, 0/1
        org_1:
            B-ORG (P, R): 1/1, 1/1
            I-ORG (P, R): 1/1, 1/1
            L-ORG (P, R): 1/1, 1/1
            U-ORG (P, R): 0/0 (nan), 0/0 (nan)
        org_2:
            B-ORG (P, R): 1/1, 1/1
            I-ORG (P, R): 0/0, 0/1
            L-ORG (P, R): 0/1, 0/1
            U-ORG (P, R): 0/1, 0/0 (nan)
        place_1:
            B-NORP (P, R): Skipped -- Not in Gold Dataset
            I-NORP (P, R): Skipped -- Not in Gold Dataset
            L-NORP (P, R): Skipped -- Not in Gold Dataset
            U-NORP (P, R): Skipped -- Not in Gold Dataset
            B-GPE (P, R): 0/0, 0/0
            I-GPE (P, R): 0/0, 0/0
            L-GPE (P, R): 0/0, 0/0
            U-GPE (P, R): 1/1, 1/1
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        analysis_corpus,
        labels,
        strict_match=True
    )
    result = lf_analysis.lf_empirical_scores(
        *analysis_corpus_y,
        agg=False,
        nan_to_num=-1
    )

    assert(result['name_1']['B-PERSON']['precision'] == 1.)
    assert(result['name_1']['B-PERSON']['recall'] == 1.)
    assert(result['name_1']['I-PERSON']['precision'] == -1.)
    assert(result['name_1']['I-PERSON']['recall'] == -1.)
    assert(result['name_1']['L-PERSON']['precision'] == 1.)
    assert(result['name_1']['L-PERSON']['recall'] == 1.)
    assert(result['name_1']['U-PERSON']['precision'] == 1.)
    assert(result['name_1']['U-PERSON']['recall'] == 1.)

    assert(result['name_2']['B-PERSON']['precision'] == 1.)
    assert(result['name_2']['B-PERSON']['recall'] == 1.)
    assert(result['name_2']['I-PERSON']['precision'] == -1.)
    assert(result['name_2']['I-PERSON']['recall'] == -1.)
    assert(result['name_2']['L-PERSON']['precision'] == 1.)
    assert(result['name_2']['L-PERSON']['recall'] == 1.)
    assert(result['name_2']['U-PERSON']['precision'] == -1.)
    assert(result['name_2']['U-PERSON']['recall'] == 0.)

    assert(result['org_1']['B-ORG']['precision'] == 1.)
    assert(result['org_1']['B-ORG']['recall'] == 1.)
    assert(result['org_1']['I-ORG']['precision'] == 1.)
    assert(result['org_1']['I-ORG']['recall'] == 1.)
    assert(result['org_1']['L-ORG']['precision'] == 1.)
    assert(result['org_1']['L-ORG']['recall'] == 1.)
    assert(result['org_1']['U-ORG']['precision'] == -1.)
    assert(result['org_1']['U-ORG']['recall'] == -1.)

    assert(result['org_2']['B-ORG']['precision'] == 1.)
    assert(result['org_2']['B-ORG']['recall'] == 1.)
    assert(result['org_2']['I-ORG']['precision'] == -1.)
    assert(result['org_2']['I-ORG']['recall'] == 0.)
    assert(result['org_2']['L-ORG']['precision'] == 0.)
    assert(result['org_2']['L-ORG']['recall'] == 0.)
    assert(result['org_2']['U-ORG']['precision'] == 0.)
    assert(result['org_2']['U-ORG']['recall'] == -1.)

    assert(result['place_1']['B-GPE']['precision'] == -1.)
    assert(result['place_1']['B-GPE']['recall'] == -1.)
    assert(result['place_1']['I-GPE']['precision'] == -1.)
    assert(result['place_1']['I-GPE']['recall'] == -1.)
    assert(result['place_1']['L-GPE']['precision'] == -1.)
    assert(result['place_1']['L-GPE']['recall'] == -1.)
    assert(result['place_1']['U-GPE']['precision'] == 1.)
    assert(result['place_1']['U-GPE']['recall'] == 1.)

    assert('B-NORP' not in result['place_1'].keys())
    assert('I-NORP' not in result['place_1'].keys())
    assert('L-NORP' not in result['place_1'].keys())
    assert('U-NORP' not in result['place_1'].keys())


def test_lf_scores_without_strict_match_with_prefixes_without_agg(
    analysis_corpus,
    analysis_corpus_y
):
    """ Test expected precision and recall scores across below spans:

    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Precision, Recall:
        name_1:
            PERSON:
                P: 3/3
                R: 3/3
        name_2:
            PERSON:
                P: 2/2
                R: 2/3
        org_1:
            ORG:
                P: 3/3 
                R: 3/3
        org_2:
            ORG:
                P: 2/3
                R: 2/3
        place_1:
            NORP: 0 (No values in test dataset)
            GPE: 
                P: 1/1
                R: 1/1
    """
    labels = ["O"]
    labels += [
        "%s-%s"%(p,l) for l in ["GPE", "NORP", "ORG", "PERSON"] for p in "BILU"
    ]
    lf_analysis = LFAnalysis(
        analysis_corpus,
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_empirical_scores(
        *analysis_corpus_y,
        agg=False
    )
    assert(result['name_1']['PERSON']['precision'] == 1.)
    assert(result['name_1']['PERSON']['recall'] == 1.)
    assert(result['name_2']['PERSON']['precision'] == 1.)
    assert(result['name_2']['PERSON']['recall'] == 2/3)
    assert(result['org_1']['ORG']['precision'] == 1.)
    assert(result['org_1']['ORG']['recall'] == 1.)
    assert(result['org_2']['ORG']['precision'] == 2/3)
    assert(result['org_2']['ORG']['recall'] == 2/3)
    assert(result['place_1']['GPE']['precision'] == 1.)
    assert(result['place_1']['GPE']['recall'] == 1.)


def test_lf_scores_without_strict_match_without_prefixes_without_agg(
    analysis_corpus,
    analysis_corpus_y
):
    """ Test expected precision and recall across below spans:

    Spans:
        "name_1": Pierre (PERSON), Lison (PERSON), Pierre(PERSON)
        "name_2": Pierre (PERSON), Lison (PERSON)
        "org_1": Norwegian (ORG), Computing (ORG), Center(ORG)
        "org_2": Norwegian (ORG), Computing (ORG), Oslo (ORG)
        "place_1": Norwegian (NORP), Oslo (GPE)

    Precision, Recall:
        name_1:
            PERSON:
                P: 3/3
                R: 3/3
        name_2:
            PERSON:
                P: 2/2
                R: 2/3
        org_1:
            ORG:
                P: 3/3 
                R: 3/3
        org_2:
            ORG:
                P: 2/3
                R: 2/3
        place_1:
            NORP: 0 (No values in test dataset)
            GPE: 
                P: 1/1
                R: 1/1
    """
    labels = ["O", "GPE", "NORP", "ORG", "PERSON"]
    lf_analysis = LFAnalysis(
        analysis_corpus,
        labels,
        strict_match=False
    )
    result = lf_analysis.lf_empirical_scores(
        *analysis_corpus_y,
        agg=False
    )
    assert(result['name_1']['PERSON']['precision'] == 1.)
    assert(result['name_1']['PERSON']['recall'] == 1.)
    assert(result['name_2']['PERSON']['precision'] == 1.)
    assert(result['name_2']['PERSON']['recall'] == 2/3)
    assert(result['org_1']['ORG']['precision'] == 1.)
    assert(result['org_1']['ORG']['recall'] == 1.)
    assert(result['org_2']['ORG']['precision'] == 2/3)
    assert(result['org_2']['ORG']['recall'] == 2/3)
    assert(result['place_1']['GPE']['precision'] == 1.)
    assert(result['place_1']['GPE']['recall'] == 1.)
