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
from spacy.tokens import Doc, Span #type: ignore

# Data files for gazetteers
WIKIDATA = os.path.dirname(__file__) + "/../../data/wikidata_tokenised.json"
WIKIDATA_SMALL =  os.path.dirname(__file__) + "/../../data/wikidata_small_tokenised.json"
COMPANY_NAMES = os.path.dirname(__file__) + "/../../data/company_names_tokenised.json"
GEONAMES =  os.path.dirname(__file__) + "/../../data/geonames.json"
CRUNCHBASE =  os.path.dirname(__file__) + "/../../data/crunchbase.json"
PRODUCTS =  os.path.dirname(__file__) + "/../../data/products.json"
FIRST_NAMES =  os.path.dirname(__file__) + "/../../data/first_names.json"
FORM_FREQUENCIES =  os.path.dirname(__file__) + "/../../data/form_frequencies.json"

# List of currency symbols and three-letter codes
CURRENCY_SYMBOLS =  {"$", "¥", "£", "€", "kr", "₽", "R$", "₹", "Rp", "₪", "zł", "Rs", "₺", "RS"}

CURRENCY_CODES = {"USD", "EUR", "CNY", "JPY", "GBP", "NOK", "DKK", "CAD", "RUB", "MXN", "ARS", "BGN", 
                  "BRL", "CHF",  "CLP", "CZK", "INR", "IDR", "ILS", "IRR", "IQD", "KRW", "KZT", "NGN", 
                  "QAR", "SEK", "SYP", "TRY", "UAH", "AED", "AUD", "COP", "MYR", "SGD", "NZD", "THB", 
                  "HUF", "HKD", "ZAR", "PHP", "KES", "EGP", "PKR", "PLN", "XAU", "VND", "GBX"}

# sets of tokens used for the shallow patterns
MONTHS = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"}
MONTHS_ABBRV =  {"Jan.", "Feb.", "Mar.", "Apr.", "May.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec."}
DAYS = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
DAYS_ABBRV = {"Mon.", "Tu.", "Tue.", "Tues.", "Wed.", "Th.", "Thu.", "Thur.", "Thurs.", "Fri.", "Sat.", "Sun."}
MAGNITUDES = {"million", "billion", "mln", "bln", "bn", "thousand", "m", "k", "b", "m.", "k.", "b.", "mln.", "bln.", "bn."}     
UNITS = {"tons", "tonnes", "barrels", "m", "km", "miles", "kph", "mph", "kg", "°C", "dB", "ft", "gal", "gallons", "g", "kW", "s", "oz",
        "m2", "km2", "yards", "W", "kW", "kWh", "kWh/yr", "Gb", "MW", "kilometers", "meters", "liters", "litres", "g", "grams", "tons/yr",
        'pounds', 'cubits', 'degrees', 'ton', 'kilograms', 'inches', 'inch', 'megawatts', 'metres', 'feet', 'ounces', 'watts', 'megabytes',
        'gigabytes', 'terabytes', 'hectares', 'centimeters', 'millimeters', "F", "Celsius"}
ORDINALS = ({"first, second, third", "fourth", "fifth", "sixth", "seventh"} | 
            {"%i1st"%i for i in range(100)} | {"%i2nd"%i for i in range(100)} | {"%ith"%i for i in range(1000)})
ROMAN_NUMERALS = {'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX','X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII',
                'XVIII', 'XIX', 'XX', 'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX'}

# Full list of country names
COUNTRIES = {'Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua', 'Argentina', 'Armenia', 'Australia', 'Austria', 
             'Azerbaijan', 'Bahamas', 'Bahrain',  'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 
             'Bolivia', 'Bosnia Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria',  'Burkina', 'Burundi', 'Cambodia', 'Cameroon', 
             'Canada', 'Cape Verde', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo', 'Costa Rica', 
             'Croatia', 'Cuba', 'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'East Timor', 
             'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Fiji', 'Finland', 'France', 
             'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece',  'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 
             'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Ivory Coast',
             'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Korea North', 'Korea South', 'Kosovo', 'Kuwait', 'Kyrgyzstan',
             'Laos','Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macedonia', 'Madagascar', 
             'Malawi', 'Malaysia', 'Maldives','Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia', 
             'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique','Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 
             'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Panama', 'Papua New Guinea',
             'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russian Federation', 'Rwanda', 'St Kitts & Nevis', 
             'St Lucia', 'Saint Vincent & the Grenadines','Samoa', 'San Marino', 'Sao Tome & Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 
             'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Sudan', 
             'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 
             'Thailand', 'Togo', 'Tonga', 'Trinidad & Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 
             'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Vatican City', 'Venezuela', 
             'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe', "USA", "UK", "Russia", "South Korea"}

# Natialities, religious and political groups
NORPS = {'Afghan', 'African', 'Albanian', 'Algerian', 'American', 'Andorran', 'Anglican', 'Angolan', 'Arab',  'Aramean','Argentine', 'Armenian', 
         'Asian', 'Australian', 'Austrian', 'Azerbaijani', 'Bahamian', 'Bahraini', 'Baklan', 'Bangladeshi', 'Batswana', 'Belarusian', 'Belgian',
         'Belizean', 'Beninese', 'Bermudian', 'Bhutanese', 'Bolivian', 'Bosnian', 'Brazilian', 'British', 'Bruneian', 'Buddhist', 
         'Bulgarian', 'Burkinabe', 'Burmese', 'Burundian', 'Californian', 'Cambodian', 'Cameroonian', 'Canadian', 'Cape Verdian', 'Catholic', 'Caymanian', 
         'Central African',  'Central American', 'Chadian', 'Chilean', 'Chinese', 'Christian', 'Christian-Democrat', 'Christian-Democratic', 
         'Colombian', 'Communist', 'Comoran', 'Congolese', 'Conservative', 'Costa Rican', 'Croat', 'Cuban', 'Cypriot', 'Czech', 'Dane',  'Danish', 
         'Democrat', 'Democratic', 'Djibouti', 'Dominican', 'Dutch', 'East European', 'Ecuadorean', 'Egyptian', 'Emirati', 'English', 'Equatoguinean',
         'Equatorial Guinean', 'Eritrean', 'Estonian', 'Ethiopian', 'Eurasian', 'European', 'Fijian', 'Filipino', 'Finn', 'Finnish', 'French', 
         'Gabonese', 'Gambian', 'Georgian', 'German', 'Germanic', 'Ghanaian', 'Greek', 'Greenlander', 'Grenadan', 'Grenadian', 'Guadeloupean', 'Guatemalan',
         'Guinea-Bissauan', 'Guinean', 'Guyanese', 'Haitian', 'Hawaiian', 'Hindu', 'Hinduist', 'Hispanic', 'Honduran', 'Hungarian', 'Icelander', 'Indian', 
         'Indonesian', 'Iranian', 'Iraqi', 'Irish', 'Islamic','Islamist', 'Israeli', 'Israelite', 'Italian', 'Ivorian', 'Jain', 'Jamaican', 'Japanese', 
         'Jew',  'Jewish', 'Jordanian', 'Kazakhstani', 'Kenyan', 'Kirghiz', 'Korean', 'Kurd', 'Kurdish',  'Kuwaiti', 'Kyrgyz', 'Labour', 'Latin',
         'Latin American', 'Latvian', 'Lebanese', 'Liberal',  'Liberian', 'Libyan', 'Liechtensteiner', 'Lithuanian', 'Londoner', 'Luxembourger', 
         'Macedonian', 'Malagasy', 'Malawian','Malaysian', 'Maldivan', 'Malian', 'Maltese', 'Manxman', 'Marshallese', 'Martinican', 'Martiniquais', 
         'Marxist', 'Mauritanian', 'Mauritian', 'Mexican', 'Micronesian', 'Moldovan', 'Mongolian', 'Montenegrin', 'Montserratian', 'Moroccan', 
         'Motswana', 'Mozambican', 'Muslim', 'Myanmarese', 'Namibian',  'Nationalist', 'Nazi', 'Nauruan', 'Nepalese', 'Netherlander', 'New Yorker',
         'New Zealander', 'Nicaraguan', 'Nigerian', 'Nordic', 'North American', 'North Korean','Norwegian','Orthodox', 'Pakistani', 'Palauan', 
         'Palestinian', 'Panamanian', 'Papua New Guinean', 'Paraguayan', 'Parisian', 'Peruvian', 'Philistine', 'Pole', 'Polish', 'Portuguese', 
         'Protestant', 'Puerto Rican', 'Qatari', 'Republican', 'Roman', 'Romanian', 'Russian', 'Rwandan', 'Saint Helenian', 'Saint Lucian',   
         'Saint Vincentian', 'Salvadoran', 'Sammarinese', 'Samoan', 'San Marinese', 'Sao Tomean', 'Saudi', 'Saudi Arabian', 'Scandinavian', 'Scottish', 
         'Senegalese', 'Serb', 'Serbian', 'Shia', 'Shiite', 'Sierra Leonean', 'Sikh', 'Singaporean', 'Slovak', 'Slovene', 'Social-Democrat', 'Socialist', 
         'Somali', 'South African', 'South American', 'South Korean', 'Soviet', 'Spaniard', 'Spanish', 'Sri Lankan', 'Sudanese', 'Sunni', 
         'Surinamer', 'Swazi', 'Swede', 'Swedish', 'Swiss', 'Syrian', 'Taiwanese', 'Tajik', 'Tanzanian', 'Taoist', 'Texan', 'Thai', 'Tibetan', 
         'Tobagonian', 'Togolese', 'Tongan', 'Tunisian', 'Turk', 'Turkish', 'Turkmen(s)', 'Tuvaluan', 'Ugandan', 'Ukrainian', 'Uruguayan', 'Uzbek', 
         'Uzbekistani', 'Venezuelan', 'Vietnamese', 'Vincentian', 'Virgin Islander', 'Welsh', 'West European', 'Western', 'Yemeni', 'Yemenite', 
         'Yugoslav', 'Zambian', 'Zimbabwean', 'Zionist'}
               
# Facilities
FACILITIES = {"Palace", "Temple", "Gate", "Museum", "Bridge", "Road", "Airport", "Hospital", "School", "Tower", "Station", "Avenue", 
             "Prison", "Building", "Plant", "Shopping Center", "Shopping Centre", "Mall", "Church", "Synagogue", "Mosque", "Harbor", "Harbour", 
              "Rail", "Railway", "Metro", "Tram", "Highway", "Tunnel", 'House', 'Field', 'Hall', 'Place', 'Freeway', 'Wall', 'Square', 'Park', 
              'Hotel'}

# Legal documents
LEGAL = {"Law", "Agreement", "Act", 'Bill', "Constitution", "Directive", "Treaty", "Code", "Reform", "Convention", "Resolution", "Regulation", 
         "Amendment", "Customs", "Protocol", "Charter"}

# event names
EVENTS = {"War", "Festival", "Show", "Massacre", "Battle", "Revolution", "Olympics", "Games", "Cup", "Week", "Day", "Year", "Series"}

# Names of languages
LANGUAGES = {'Afar', 'Abkhazian', 'Avestan', 'Afrikaans', 'Akan', 'Amharic', 'Aragonese', 'Arabic', 'Aramaic', 'Assamese', 'Avaric', 'Aymara', 
             'Azerbaijani', 'Bashkir',  'Belarusian', 'Bulgarian', 'Bambara', 'Bislama', 'Bengali', 'Tibetan', 'Breton', 'Bosnian', 'Cantonese', 
             'Catalan', 'Chechen',  'Chamorro', 'Corsican', 'Cree', 'Czech', 'Chuvash', 'Welsh',  'Danish', 'German', 'Divehi', 'Dzongkha', 'Ewe', 
             'Greek', 'English', 'Esperanto', 'Spanish', 'Castilian',  'Estonian', 'Basque', 'Persian', 'Fulah', 'Filipino', 'Finnish', 'Fijian', 'Faroese', 
             'French', 'Western Frisian', 'Irish', 'Gaelic', 'Galician', 'Guarani', 'Gujarati', 'Manx', 'Hausa', 'Hebrew', 'Hindi', 'Hiri Motu', 
             'Croatian', 'Haitian', 'Hungarian', 'Armenian', 'Herero', 'Indonesian', 'Igbo', 'Inupiaq', 'Ido', 'Icelandic', 'Italian', 'Inuktitut', 
             'Japanese', 'Javanese', 'Georgian', 'Kongo', 'Kikuyu', 'Kuanyama', 'Kazakh', 'Kalaallisut', 'Greenlandic', 'Central Khmer', 'Kannada', 
             'Korean', 'Kanuri', 'Kashmiri', 'Kurdish','Komi', 'Cornish', 'Kirghiz', 'Latin', 'Luxembourgish', 'Ganda', 'Limburgish', 'Lingala', 'Lao', 
             'Lithuanian', 'Luba-Katanga', 'Latvian', 'Malagasy', 'Marshallese', 'Maori', 'Macedonian', 'Malayalam', 'Mongolian', 'Marathi', 'Malay', 
             'Maltese', 'Burmese', 'Nauru', 'Bokmål', 'Norwegian', 'Ndebele', 'Nepali', 'Ndonga', 'Dutch', 'Flemish', 'Nynorsk', 'Navajo', 'Chichewa', 
             'Occitan', 'Ojibwa', 'Oromo', 'Oriya', 'Ossetian', 'Punjabi', 'Pali', 'Polish', 'Pashto', 'Portuguese', 'Quechua', 'Romansh', 'Rundi', 
             'Romanian', 'Russian', 'Kinyarwanda', 'Sanskrit', 'Sardinian', 'Sindhi', 'Sami', 'Sango', 'Sinhalese',  'Slovak', 'Slovenian', 'Samoan', 
             'Shona', 'Somali', 'Albanian', 'Serbian', 'Swati', 'Sotho', 'Sundanese', 'Swedish', 'Swahili', 'Tamil', 'Telugu', 'Tajik', 'Thai', 
             'Tigrinya', 'Turkmen', 'Taiwanese', 'Tagalog', 'Tswana', 'Tonga', 'Turkish', 'Tsonga', 'Tatar', 'Twi', 'Tahitian', 'Uighur', 'Ukrainian', 
             'Urdu', 'Uzbek', 'Venda', 'Vietnamese', 'Volapük', 'Walloon', 'Wolof', 'Xhosa', 'Yiddish', 'Yoruba', 'Zhuang', 'Mandarin', 
             'Mandarin Chinese',  'Chinese', 'Zulu'}


LEGAL_SUFFIXES = {
    'ltd',     # Limited ~13.000
    'llc',     # limited liability company (UK)
    'ltda',    # limitada (Brazil, Portugal)
    'inc',     # Incorporated ~9700
    'co ltd',  # Company Limited ~9200
    'corp',    # Corporation ~5200
    'sa',      # Spółka Akcyjna (Poland), Société Anonyme (France)  ~3200
    'plc',     # Public Limited Company (Great Britain) ~2100
    'ag',      # Aktiengesellschaft (Germany) ~1000
    'gmbh',    # Gesellschaft mit beschränkter Haftung  (Germany)
    'bhd',     # Berhad (Malaysia) ~900
    'jsc',     # Joint Stock Company (Russia) ~900
    'co',      # Corporation/Company ~900
    'ab',      # Aktiebolag (Sweden) ~800
    'ad',      # Akcionarsko Društvo (Serbia), Aktsionerno Drujestvo (Bulgaria) ~600
    'tbk',     # Terbuka (Indonesia) ~500
    'as',      # Anonim Şirket (Turkey), Aksjeselskap (Norway) ~500
    'pjsc',    # Public Joint Stock Company (Russia, Ukraine) ~400
    'spa',     # Società Per Azioni (Italy) ~300
    'nv',      # Naamloze vennootschap (Netherlands, Belgium) ~230
    'dd',      # Dioničko Društvo (Croatia) ~220
    'a s',     # a/s (Denmark), a.s (Slovakia) ~210
    'oao',     # Открытое акционерное общество (Russia) ~190
    'asa',     # Allmennaksjeselskap (Norway) ~160
    'ojsc',    # Open Joint Stock Company (Russia) ~160
    'lp',      # Limited Partnership (US) ~140
    'llp',     # limited liability partnership
    'oyj',     # julkinen osakeyhtiö (Finland) ~120
    'de cv',   # Capital Variable (Mexico) ~120
    'se',      # Societas Europaea (Germany) ~100
    'kk',      # kabushiki gaisha (Japan)
    'aps',     # Anpartsselskab (Denmark)
    'cv',      # commanditaire vennootschap (Netherlands)
    'sas',     # société par actions simplifiée (France)
    'sro',     # Spoločnosť s ručením obmedzeným (Slovakia)
    'oy',      # Osakeyhtiö (Finland)
    'kg',      # Kommanditgesellschaft (Germany)
    'bv',      # Besloten Vennootschap (Netherlands)
    'sarl',    # société à responsabilité limitée (France)
    'srl',     # Società a responsabilità limitata (Italy)
    'sl'       # 	Sociedad Limitada (Spain) 
}
# Generic words that may appear in official company names but are sometimes skipped when mentioned in news articles (e.g. Nordea Bank -> Nordea)
GENERIC_TOKENS = {"International", "Group", "Solutions", "Technologies", "Management", "Association", "Associates", "Partners", 
                  "Systems", "Holdings", "Services", "Bank", "Fund",  "Stiftung", "Company"}

# List of tokens that are typically lowercase even when they occur in capitalised segments (e.g. International Council of Shopping Centers)
LOWERCASED_TOKENS = {"'s", "-", "a", "an", "the", "at", "by", "for", "in", "of", "on", "to", "up", "and", "&"}

# Prefixes to family names that are often in lowercase
NAME_PREFIXES = {"-", "von", "van", "de", "di", "le", "la", "het", "'t'", "dem", "der", "den", "d'", "ter"}


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
        print("Loading NER models")
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
        proper_detector = TokenConstraintAnnotator("proper_detector", lambda tok: utils.is_likely_proper(tok), "ENT")
    
        # Detection based on casing, but allowing some lowercased tokens
        proper2_detector = TokenConstraintAnnotator("proper2_detector", lambda tok: utils.is_likely_proper(tok), "ENT")
        proper2_detector.add_gap_tokens(LOWERCASED_TOKENS | NAME_PREFIXES)
        
        # Detection based on part-of-speech tags
        nnp_detector = TokenConstraintAnnotator("nnp_detector", lambda tok: tok.tag_=="NNP", "ENT")
        
        # Detection based on dependency relations (compound phrases)
        compound = lambda tok: utils.is_likely_proper(tok) and utils.in_compound(tok)
        compound_detector = TokenConstraintAnnotator("compound_detector", compound, "ENT")
 
        exclusives = ["date_detector", "time_detector", "money_detector"]
        for annotator in [proper_detector, proper2_detector, nnp_detector, compound_detector]:
            annotator.add_incompatible_sources(exclusives)
            annotator.add_gap_tokens(["'s", "-"])
            self.add_annotator(annotator)

            # We add one variants for each NE detector, looking at infrequent tokens
            infrequent_name = "infrequent_%s"%annotator.name
            self.add_annotator(SpanConstraintAnnotator(infrequent_name, annotator.name, utils.is_infrequent))
        
        # Other types (legal references etc.)      
        misc_detector = FunctionAnnotator("misc_detector", misc_generator)
        legal_detector = FunctionAnnotator("legal_detector", legal_generator)
        
        # Detection of companies with a legal type
        ends_with_legal_suffix = lambda x: x[-1].lower_.rstrip(".") in LEGAL_SUFFIXES
        company_type_detector = SpanConstraintAnnotator("company_type_detector", "proper2_detector", 
                                                        ends_with_legal_suffix, "COMPANY")

        # Detection of person names
        full_name_detector = SpanConstraintAnnotator("full_name_detector", "proper2_detector", 
                                                     FullNameDetector(), "PERSON")
        name_detector2 = SpanConstraintAnnotator("name_detector", "proper_detector", 
                                                 constraint=name_detector, label="PERSON")
        
        for annotator in [misc_detector, legal_detector, company_type_detector, 
                          full_name_detector, name_detector2]:
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
        
        self.add_annotator(ModelAnnotator("conll2003", os.path.dirname(__file__) 
                                          + "/../../data/conll2003"))
        self.add_annotator(TruecaseAnnotator("conll2003_truecase", os.path.dirname(__file__) + 
                                             "/../../data/conll2003", FORM_FREQUENCIES))
        self.add_annotator(ModelAnnotator("BTC", os.path.dirname(__file__) + "/../../data/btc"))
        self.add_annotator(TruecaseAnnotator("BTC_truecase", os.path.dirname(__file__) + 
                                             "/../../data/btc", FORM_FREQUENCIES))

        # Avoid spans that start with an article
        editor = lambda span: span[1:] if span[0].lemma_ in {"the", "a", "an"} else span
        self.add_annotator(SpanEditorAnnotator("edited_BTC", "BTC", editor))
        self.add_annotator(SpanEditorAnnotator("edited_BTC_truecase", "BTC_truecase", editor))
        self.add_annotator(SpanEditorAnnotator("edited_conll2003", "conll2003", editor))
        self.add_annotator(SpanEditorAnnotator("edited_conll2003_truecase", "conll2003_truecase", editor))


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
        for name, tries in {"wiki":wiki_tries, "wiki_small":wiki_small_tries, "geo":geo_tries, 
                            "crunchbase":crunchbase_tries, "products":products_tries}.items():
            
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
        
        self.add_annotator(Muc6Standardiser())
        
        maj_voter = MajorityVoter("doclevel_voter", ["LOCATION", "ORGANIZATION", "PERSON"])
        maj_voter.sources_to_avoid = ["doc_history", "doc_majority"]
        maj_voter.add_underspecified_label("ENT", {"LOCATION", "ORGANIZATION", "PERSON"})     
        self.add_annotator(maj_voter)   
           
        self.add_annotator(DocumentHistoryAnnotator("doc_history_cased", "doclevel_voter", ["PERSON", "ORGANIZATION"]))
        self.add_annotator(DocumentHistoryAnnotator("doc_history_uncased", "doclevel_voter", ["PERSON", "ORGANIZATION"],
                                                    case_sentitive=False))
        
        maj_voter = MajorityVoter("doclevel_voter", ["LOCATION", "ORGANIZATION", "PERSON"])
        maj_voter.sources_to_avoid = ["doc_majority"]
        maj_voter.add_underspecified_label("ENT", {"LOCATION", "ORGANIZATION", "PERSON"})     
        self.add_annotator(maj_voter)   

        self.add_annotator(DocumentMajorityAnnotator("doc_majority_cased", "doclevel_voter"))
        self.add_annotator(DocumentMajorityAnnotator("doc_majority_uncased", "doclevel_voter", case_sensitive=False))
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
        if tok.lemma_ in DAYS | DAYS_ABBRV:
            spans.append((i,i+1, "DATE"))
        elif tok.is_digit and re.match("\\d+$", tok.text) and int(tok.text) > 1920 and int(tok.text) < 2040:
            spans.append((i,i+1, "DATE"))
        elif tok.lemma_ in MONTHS | MONTHS_ABBRV:       
            if tok.tag_=="MD": # Skipping "May" used as auxiliary
                pass
            elif i > 0 and re.match("\\d+$", doc[i-1].text) and int(doc[i-1].text) < 32:
                spans.append((i-1,i+1, "DATE"))
            elif i > 1 and re.match("\\d+(?:st|nd|rd|th)$", doc[i-2].text) and doc[i-1].lower_=="of":
                spans.append((i-2,i+1, "DATE"))
            elif i < len(doc)-1 and re.match("\\d+$", doc[i+1].text) and int(doc[i+1].text) < 32: 
                spans.append((i,i+2, "DATE"))
                i += 1
            else:
                spans.append((i,i+1, "DATE"))
        i += 1

    for start,end, content in utils.merge_contiguous_spans(spans, doc):
            yield start, end, content
                
             
    
def time_generator(doc):
    """Searches for occurrences of time patterns in text"""
    
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
    
    
       
def money_generator(doc):
    """Searches for occurrences of money patterns in text"""      
        
    i = 0
    while i < len(doc):
        tok = doc[i]
        if tok.text[0].isdigit():
            j = i+1
            while (j < len(doc) and (doc[j].text[0].isdigit() or  doc[j].norm_ in MAGNITUDES)):
                j += 1
                
            found_symbol = False
            if i > 0 and doc[i-1].text in (CURRENCY_CODES | CURRENCY_SYMBOLS):
                i = i-1
                found_symbol = True
            if (j < len(doc) and doc[j].text in 
                (CURRENCY_CODES | CURRENCY_SYMBOLS | {"euros", "cents", "rubles"})):
                j += 1
                found_symbol = True
                
            if found_symbol:
                yield i,j, "MONEY"
            i = j
        else:
            i += 1
            
    
    
def number_generator(doc):
    """Searches for occurrences of number patterns (cardinal, ordinal, quantity or percent) in text"""

   
    i = 0
    while i < len(doc):
        tok = doc[i]
    
        if tok.lower_ in ORDINALS:
            yield i, i+1, "ORDINAL"
            
        elif re.search("\\d", tok.text):
            j = i+1
            while (j < len(doc) and (doc[j].norm_ in MAGNITUDES)):
                j += 1
            if j < len(doc) and doc[j].lower_.rstrip(".") in UNITS:
                j += 1
                yield i, j, "QUANTITY"
            elif j < len(doc) and doc[j].lower_ in ["%", "percent", "pc.", "pc", "pct", "pct.", "percents", "percentage"]:
                j += 1
                yield i, j, "PERCENT"        
            else:
                yield i, j,  "CARDINAL"
            i = j-1
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
        
        if span[0].text in self.first_names and span[-1].is_alpha and span[-1].is_title:
           return True
        elif (span[0].text.endswith(".") and len(span)>1 and span[1].text in self.first_names 
              and span[-1].is_alpha and span[-1].is_title):
           return True
        return False
       
       
def name_detector(span: Span) -> bool: 
    """Search for names that have a Mr/Mrs/Miss/Dr/Sen in front"""
    
    if span.start==0 or len(span) > 5 or not span[-1].is_alpha or not span[-1].is_title or len(span)==1:
        return False
    
    return span.doc[span.start].text.rstrip(".") in {"Mr", "Mrs", "Miss", "Dr", "Sen"}  
                    

class SnipsAnnotator(SpanAnnotator):
    """Annotation using the Snips NLU entity parser. """
    
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
            if (result["entity_kind"]=="snips/number" and span.text.lower() not in 
                {"one", "some", "few", "many", "several"}):
                label = "CARDINAL"
            elif (result["entity_kind"]=="snips/ordinal" and span.text.lower() not in 
                  {"first", "second", "the first", "the second"}):
                label = "ORDINAL"
            elif result["entity_kind"]=="snips/temperature":
                label = "QUANTITY"
            elif result["entity_kind"]=="snips/amountOfMoney":
                label = "MONEY"
            elif result["entity_kind"]=="snips/percentage":
                label = "PERCENT"
            elif result["entity_kind"] in {"snips/date", "snips/datePeriod", "snips/datetime"}:
                label = "DATE"
            elif result["entity_kind"] in {"snips/time", "snips/timePeriod"}:
                label = "TIME"
            
            if label in {"DATE", "TIME"} and span[0].text in {"from", "to", "in", "since", "at", "on"}:
                yield (span.start+1), span.end, label  
            elif label: 
                yield span.start, span.end, label

                
def legal_generator(doc):
   
    legal_spans = []
    for span in utils.get_spans(doc, ["proper2_detector", "nnp_detector"]):
        if not utils.is_likely_proper(doc[span.end-1]):
            continue         
        last_token = doc[span.end-1].text.title().rstrip("s")
                  
        if last_token in LEGAL:     
            legal_spans.append((span.start,span.end, "LAW"))
                     
    
    # Handling legal references such as Article 5
    for i in range(len(doc)-1):
        if doc[i].text.rstrip("s") in {"Article", "Paragraph", "Section", "Chapter", "§"}:
            if doc[i+1].text[0].isdigit() or doc[i+1].text in ROMAN_NUMERALS:
                start, end = i, i+2
                if (i < len(doc)-3 and doc[i+2].text in {"-", "to", "and"} 
                    and (doc[i+3].text[0].isdigit() or doc[i+3].text in ROMAN_NUMERALS)):
                    end = i+4
                legal_spans.append((start,end, "LAW"))

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

        if span_text in COUNTRIES:
            yield span.start, span.end, "GPE"

        if len(span) <= 3 and (span in NORPS or last_token in NORPS or last_token.rstrip("s") in NORPS):
            yield span.start, span.end, "NORP"
    
        if span in LANGUAGES and doc[span.start].tag_=="NNP":
            yield span.start, span.end, "LANGUAGE"
            
        if last_token in FACILITIES and len(span) > 1:
            yield span.start, span.end, "FAC"     

        if last_token in EVENTS  and len(span) > 1:
            yield span.start, span.end, "EVENT"     
    
       
    
############################################
# Standardisation of the output labels
############################################    


class Muc6Standardiser(SpanAnnotator):
    """Annotator taking existing annotations and standardising them 
    to fit the ConLL 2003 tag scheme"""

    def __init__(self):
        super(Muc6Standardiser,self).__init__("")
        
       
    def __call__(self, doc):
        """Annotates one single document"""     
               
        for source in doc.spans:
               
            new_spans = []  
            for span in doc.spans[source]:

                if (span.label_ in {"DATE", "TIME"} and not re.search("\\d", span.text) 
                    and not any(tok.text in MONTHS for tok in span)):
                    continue
                elif span.label_=="PER":
                    new_spans.append(Span(doc, span.start, span.end, label="PERSON"))
                elif span.label_ in {"ORG", "ORGANISATION", "COMPANY"}:
                    new_spans.append(Span(doc, span.start, span.end, label="ORGANIZATION"))
                elif span.label_ in {"LOC", "GPE"}:
                    new_spans.append(Span(doc, span.start, span.end, label="LOCATION"))
     #           elif span.label_ in {"EVENT", "FAC", "LANGUAGE", "LAW", "NORP", "PRODUCT", "WORK_OF_ART"}:
     #               new_spans.append(Span(doc, span.start, span.end, label="MISC"))
                else:
                    new_spans.append(span)  
                    
            # Small fix for MUC-6, which (to the opposite of most NER corpora) does
            # not include titles in the span for a person name            
            new_spans2 = []
            for span in new_spans:
                if (span.label_ in {"ENT","PERSON"} and span[0].text.rstrip(".").lower() 
                    in {"mr", "mrs", "miss", "dr", "sen"}):
                    if len(span) > 1:
                        new_spans2.append(Span(doc, span.start+1, span.end, span.label_))
                else:
                    new_spans2.append(span)
                    
            doc.spans[source] = new_spans2
                         
        return doc

