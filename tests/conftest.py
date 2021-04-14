
import pytest
import spacy

@pytest.fixture(scope="session")
def nlp():
    import spacy
    return spacy.load("en_core_web_md")

@pytest.fixture(scope="session")
def nlp_small():
    import spacy
    return spacy.load("en_core_web_sm")