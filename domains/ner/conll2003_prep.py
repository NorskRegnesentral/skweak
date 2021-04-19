
from .conll2003_ner import (WIKIDATA, WIKIDATA_SMALL, CRUNCHBASE, PRODUCTS, 
                            GEONAMES, COMPANY_NAMES)

from . import data_utils
import pickle, re, json
import spacy

"""Contains scripts used to compile the lists of entities from Wikipedia, Geonames,
Crunchbase and DBPedia. Those scripts can be ignored in most cases, as it is easier
to directly rely on the already compiled json files. """


############################################
# Compilation of data sources
############################################
   
    
def compile_wikidata(wikidata="../data/WikidataNE_20170320_NECKAR_1_0.json_.gz", only_with_descriptions=False):
    """Compiles a JSON file with the wiki data"""
     
    
    import gzip, json
    fd = gzip.open(wikidata)
    wikidata = {"PERSON":{}, "LOC":{}, "GPE":{}, "ORG":{}}
    location_qs = set()
    for l in fd:
        d = json.loads(l)
        neClass = str(d["neClass"])
        name = d["norm_name"]
        if ("en_sitelink" not in d and neClass !="PER"):
            continue
        if "en_sitelink" in d:
            if "," in d["en_sitelink"] or "(" in d["en_sitelink"]:
                continue
        if name[0].isdigit() or name[-1].isdigit() or len(name) < 2:
            continue
        if neClass=="PER":
            neClass = "PERSON"
        elif neClass=="LOC":
            if {'Mountain Range', 'River', 'Sea', 'Continent', 'Mountain'}.intersection(d.get("location_type",set())):
                neClass = "LOC"
            else:
                neClass ="GPE"
            location_qs.add(d["id"])
        elif neClass=="ORG" and d["id"] in location_qs:
            continue
        if "alias" in d:
            d["nb_aliases"] = len(d["alias"])
            del d["alias"]
        for key_to_remove in ["de_sitelink", '$oid', "id", "coordinate", "official_website", "_id"]:
            if key_to_remove in d:
                del d[key_to_remove]
        if name in wikidata[neClass]:
            merge = wikidata[neClass][name] if len(str(wikidata[neClass][name])) > len(str(d)) else d
            merge["nb_entities"] = wikidata[neClass][name].get("nb_entities", 1) + 1
            wikidata[neClass][name] = merge
        else:
            wikidata[neClass][name] = d
                  
    fd = open("data/frequencies.pkl", "rb")
    frequencies = pickle.load(fd)
    fd.close() 
    
    # We only keep entities with a certain frequency
    for neClass in ["PERSON", "LOC", "ORG", "GPE"]:
        for entity in list(wikidata[neClass].keys()): 
            if entity.lower() in frequencies and frequencies[entity.lower()]>10000: 
                del wikidata[neClass][entity]
    
    # And prune those that cannot be encoded using latin characters
    for neClass in ["PERSON", "LOC", "ORG", "GPE"]:
        for entity in list(wikidata[neClass].keys()): 
            try:
                entity.encode('iso-8859-15') 
            except UnicodeEncodeError: 
                del wikidata[neClass][entity]

        
    wikidata2 = {neClass:{} for neClass in wikidata}
    for neClass in wikidata:
        entities_for_class = set()
        for entity in wikidata[neClass]:
            nb_tokens = len(entity.split())
            if nb_tokens > 10:
                continue
            if only_with_descriptions and "description" not in wikidata[neClass][entity]:
                continue
            entities_for_class.add(entity) 
            if "en_sitelink" in wikidata[neClass][entity]:
                entities_for_class.add(wikidata[neClass][entity]["en_sitelink"])
        wikidata2[neClass] = entities_for_class #type: ignore
                            
    fd = open(WIKIDATA_SMALL if only_with_descriptions else WIKIDATA, "w")
    json.dump({key:sorted(names) for key,names in wikidata2.items()}, fd)
    fd.close()

    
def get_alternative_company_names(name, vocab=None):
    """Extract a list of alternative company names (with or without legal suffix etc.)"""
    
    alternatives = {name}        
    while True:
        current_nb_alternatives = len(alternatives)
            
        for alternative in list(alternatives):
            tokens = alternative.split()
            if len(tokens)==1:
                continue
                
            # We add an alternative name without the legal suffix
            if tokens[-1].lower().rstrip(".") in data_utils.LEGAL_SUFFIXES: 
                alternatives.add(" ".join(tokens[:-1]))
            
            if tokens[-1].lower() in {"limited", "corporation"}:
                alternatives.add(" ".join(tokens[:-1]))                
                
            if tokens[-1].lower().rstrip(".") in {"corp", "inc", "co"}:
                if alternative.endswith("."):
                    alternatives.add(alternative.rstrip("."))
                else:
                    alternatives.add(alternative+".")
                
            # If the last token is a country name (like The SAS Group Norway), add an alternative without
            if tokens[-1] in data_utils.COUNTRIES:
                alternatives.add(" ".join(tokens[:-1])) 
                
            # If the name starts with a the, add an alternative without it
            if tokens[0].lower()=="the":   
                alternatives.add(" ".join(tokens[1:]))
                
            # If the name ends with a generic token such as "Telenor International", add an alternative without
            if vocab is not None and tokens[-1] in data_utils.GENERIC_TOKENS and any([tok for tok in tokens if vocab[tok].rank==0]):
                alternatives.add(" ".join(tokens[:-1])) 
                    
        if len(alternatives)==current_nb_alternatives:
            break
    
    # We require the alternatives to have at least 2 characters (4 characters if the name does not look like an acronym)
    alternatives = {alt for alt in alternatives if len(alt) > 1 and alt.lower().rstrip(".") not in data_utils.LEGAL_SUFFIXES} 
    alternatives = {alt for alt in alternatives if len(alt) > 3 or alt.isupper()}
    
    return alternatives


def compile_company_names():
    """Compiles a JSON file with company names"""
    
    vocab = spacy.load("en_core_web_md").vocab
    
    fd = open("../data/graph/entity.sql.json")
    company_entities = set()
    other_org_entities = set()
    for l in fd:
        dico = json.loads(l)
        if ("factset_entity_type_description" not in dico or dico["factset_entity_type_description" ] not in 
            {"Private Company", "Subsidiary", "Extinct", "Public Company", "Holding Company", "College/University", 
             "Government", "Non-Profit Organization", "Operating Division", "Foundation/Endowment"}):
            continue
        name = dico["factset_entity_name"]
        name = name.split("(")[0].split(",")[0].strip(" \n\t/")
        if not name:
            continue

        alternatives = get_alternative_company_names(name, vocab)
        if dico["factset_entity_type_description" ] in {"College/University", "Government", "Non-Profit Organization", "Foundation/Endowment"}:
            other_org_entities.update(alternatives)
        else:
            company_entities.update(alternatives)
    fd.close()
    print("Number of extracted entities: %i companies and %i other organisations"%(len(company_entities), len(other_org_entities)))
    fd = open(COMPANY_NAMES, "w")
    json.dump({"COMPANY":sorted(company_entities), "ORG":sorted(other_org_entities)}, fd)
    fd.close()
    
    
def compile_geographical_data(geo_source="../data/allCountries.txt", population_threshold=100000):
    """Compiles a JSON file with geographical locations"""
    
    names = set()
    fd = open(geo_source)
    for i, line in enumerate(fd):
        line_feats = line.split("\t")
        if len(line_feats) < 15:
            continue
        population = int(line_feats[14])
        if population < population_threshold:
            continue
        name = line_feats[1].strip()
        names.add(name)
        name = re.sub(".*(?:Kingdom|Republic|Province|State|Commonwealth|Region|City|Federation) of ", "", name).strip()
        names.add(name)
        name = name.replace(" City", "").replace(" Region", "").replace(" District", "").replace(" County", "").replace(" Zone", "").strip()
        names.add(name)
        name = (name.replace("Arrondissement de ", "").replace("Stadtkreis ", "").replace("Landkreis ", "").strip()
                .replace("Departamento de ", "").replace("Département de ", "").replace("Provincia di ", "")).strip()
        names.add(name)
        name = re.sub("^the ", "", name).strip()
        names.add(name)
        if i%10000==0:
            print("Number of processed lines:", i, "and number of extracted locations:", len(names))
    fd.close()
    names = {alt for alt in names if len(alt) > 2 and alt.lower().rstrip(".") not in data_utils.LEGAL_SUFFIXES}
    fd = open(GEONAMES, "w")
    json.dump({"GPE":sorted(names)}, fd)
    fd.close()
        
        
def compile_crunchbase_data(org_data="../data/organizations.csv", people_data="../data/people.csv"):
    """Compiles a JSON file with company and person names from Crunchbase Open Data"""

    company_entities = set()
    other_org_entities = set()
    
    vocab = spacy.load("en_core_web_md").vocab
    
    fd = open(org_data)
    for line in fd:
        split = [s.strip() for s in line.rstrip().strip("\"").split("\",\"")]
        if len(split) < 5:
            continue
        name = split[1]
        alternatives = get_alternative_company_names(name, vocab)        
        if split[3] in {"company", "investor"}:
            company_entities.update(alternatives)
        else:
            other_org_entities.update(alternatives)
    fd.close()
    print("Number of extracted entities: %i companies and %i other organisations"%(len(company_entities), len(other_org_entities)))

    persons = set()
    fd = open(people_data)
    for line in fd:
        split = [s.strip() for s in line.rstrip().strip("\"").split("\",\"")]
        if len(split) < 5:
            continue
        first_name = split[2]
        last_name = split[3]
        alternatives = {"%s %s"%(first_name, last_name)}
    #    alternatives.add(last_name)
        alternatives.add("%s. %s"%(first_name[0], last_name))
        if " " in first_name:
            first_split = first_name.split(" ", 1)
            alternatives.add("%s %s"%(first_split[0], last_name))
            alternatives.add("%s %s. %s"%(first_split[0], first_split[1][0], last_name))
            alternatives.add("%s. %s. %s"%(first_split[0][0], first_split[1][0], last_name))
        persons.update(alternatives)
        
    # We require person names to have at least 3 characters (and not be a suffix)
    persons = {alt for alt in persons if len(alt) > 2 and alt.lower().rstrip(".") not in data_utils.LEGAL_SUFFIXES}
    fd.close()
    print("Number of extracted entities: %i person names"%(len(persons)))
   
    fd = open(CRUNCHBASE, "w")
    json.dump({"COMPANY":sorted(company_entities), "ORG":sorted(other_org_entities), "PERSON":sorted(persons)}, fd)
    fd.close()
    
def compile_product_data(data="../data/dbpedia.json"):
    fd = open(data)
    all_product_names = set()
    for line in fd:
        line = line.strip().strip(",")
        value = json.loads(line)["label2"]["value"]
        if "(" in value:
            continue
            
        product_names = {value}
        
        # The DBpedia entries are all titled, which cause problems for products such as iPad
        if len(value)>2 and value[0] in {"I", "E"} and value[1].isupper() and value[2].islower():
            product_names.add(value[0].lower()+value[1:])
        
        # We also add plural entries
        for product_name in list(product_names):
            if len(product_name.split()) <= 2:
                plural = product_name + ("es" if value.endswith("s") else "s")
                product_names.add(plural)
                
        all_product_names.update(product_names)
        
    fd = open(PRODUCTS, "w")
    json.dump({"PRODUCT":sorted(all_product_names)}, fd)
    fd.close()
        
        
def compile_wiki_product_data(data="../data/wiki_products.json"):
    fd = open(data)
    dict_list = json.load(fd)
    fd.close()
    products = set()
    for product_dict in dict_list:
        product_name = product_dict["itemLabel"]
        if "("  in product_name or len(product_name) <= 2:
            continue
        products.add(product_name)
        if len(product_name.split()) <= 2:
            plural = product_name + ("es" if product_name.endswith("s") else "s")
            products.add(plural)

    fd = open(WIKIDATA, "r")
    current_dict = json.load(fd)
    fd.close()
    current_dict["PRODUCT"] = sorted(products)
    fd = open(WIKIDATA, "w")
    json.dump(current_dict, fd)
    fd.close()
    
    fd = open(WIKIDATA_SMALL, "r")
    current_dict = json.load(fd)
    fd.close()
    current_dict["PRODUCT"] = sorted(products)
    fd = open(WIKIDATA_SMALL, "w")
    json.dump(current_dict, fd)
    fd.close()
