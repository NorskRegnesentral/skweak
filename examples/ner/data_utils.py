
"""Class containing some generic entity names (in English)"""

# List of currency symbols and three-letter codes
CURRENCY_SYMBOLS = {"$", "¥", "£", "€", "kr", "₽", "R$", "₹", "Rp", "₪", "zł", "Rs", "₺", "RS"}

CURRENCY_CODES = {"USD", "EUR", "CNY", "JPY", "GBP", "NOK", "DKK", "CAD", "RUB", "MXN", "ARS", "BGN",
                  "BRL", "CHF", "CLP", "CZK", "INR", "IDR", "ILS", "IRR", "IQD", "KRW", "KZT", "NGN",
                  "QAR", "SEK", "SYP", "TRY", "UAH", "AED", "AUD", "COP", "MYR", "SGD", "NZD", "THB",
                  "HUF", "HKD", "ZAR", "PHP", "KES", "EGP", "PKR", "PLN", "XAU", "VND", "GBX"}

# sets of tokens used for the shallow patterns
MONTHS = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December"}
MONTHS_ABBRV = {"Jan.", "Feb.", "Mar.", "Apr.", "May.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec."}
DAYS = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
DAYS_ABBRV = {"Mon.", "Tu.", "Tue.", "Tues.", "Wed.", "Th.", "Thu.", "Thur.", "Thurs.", "Fri.", "Sat.", "Sun."}
MAGNITUDES = {"million", "billion", "mln", "bln", "bn", "thousand", "m", "k", "b", "m.", "k.", "b.", "mln.", "bln.",
              "bn."}
UNITS = {"tons", "tonnes", "barrels", "m", "km", "miles", "kph", "mph", "kg", "°C", "dB", "ft", "gal", "gallons", "g",
         "kW", "s", "oz",
         "m2", "km2", "yards", "W", "kW", "kWh", "kWh/yr", "Gb", "MW", "kilometers", "meters", "liters", "litres", "g",
         "grams", "tons/yr",
         'pounds', 'cubits', 'degrees', 'ton', 'kilograms', 'inches', 'inch', 'megawatts', 'metres', 'feet', 'ounces',
         'watts', 'megabytes',
         'gigabytes', 'terabytes', 'hectares', 'centimeters', 'millimeters', "F", "Celsius"}
ORDINALS = ({"first, second, third", "fourth", "fifth", "sixth", "seventh"} |
            {"%i1st" % i for i in range(100)} | {"%i2nd" % i for i in range(100)} | {"%ith" % i for i in range(1000)})
ROMAN_NUMERALS = {'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI',
                  'XVII',
                  'XVIII', 'XIX', 'XX', 'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX'}

# Full list of country names
COUNTRIES = {'Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua', 'Argentina', 'Armenia', 'Australia',
             'Austria',
             'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin',
             'Bhutan',
             'Bolivia', 'Bosnia Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina', 'Burundi',
             'Cambodia', 'Cameroon',
             'Canada', 'Cape Verde', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros',
             'Congo', 'Costa Rica',
             'Croatia', 'Cuba', 'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic',
             'East Timor',
             'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Fiji',
             'Finland', 'France',
             'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea',
             'Guinea-Bissau', 'Guyana',
             'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel',
             'Italy', 'Ivory Coast',
             'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Korea North', 'Korea South', 'Kosovo',
             'Kuwait', 'Kyrgyzstan',
             'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg',
             'Macedonia', 'Madagascar',
             'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico',
             'Micronesia',
             'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru',
             'Nepal', 'Netherlands',
             'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Panama',
             'Papua New Guinea',
             'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russian Federation',
             'Rwanda', 'St Kitts & Nevis',
             'St Lucia', 'Saint Vincent & the Grenadines', 'Samoa', 'San Marino', 'Sao Tome & Principe', 'Saudi Arabia',
             'Senegal', 'Serbia',
             'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia',
             'South Africa', 'South Sudan',
             'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria', 'Taiwan',
             'Tajikistan', 'Tanzania',
             'Thailand', 'Togo', 'Tonga', 'Trinidad & Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda',
             'Ukraine',
             'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu',
             'Vatican City', 'Venezuela',
             'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe', "USA", "UK", "Russia", "South Korea"}

# Natialities, religious and political groups
NORPS = {'Afghan', 'African', 'Albanian', 'Algerian', 'American', 'Andorran', 'Anglican', 'Angolan', 'Arab', 'Aramean',
         'Argentine', 'Armenian',
         'Asian', 'Australian', 'Austrian', 'Azerbaijani', 'Bahamian', 'Bahraini', 'Baklan', 'Bangladeshi', 'Batswana',
         'Belarusian', 'Belgian',
         'Belizean', 'Beninese', 'Bermudian', 'Bhutanese', 'Bolivian', 'Bosnian', 'Brazilian', 'British', 'Bruneian',
         'Buddhist',
         'Bulgarian', 'Burkinabe', 'Burmese', 'Burundian', 'Californian', 'Cambodian', 'Cameroonian', 'Canadian',
         'Cape Verdian', 'Catholic', 'Caymanian',
         'Central African', 'Central American', 'Chadian', 'Chilean', 'Chinese', 'Christian', 'Christian-Democrat',
         'Christian-Democratic',
         'Colombian', 'Communist', 'Comoran', 'Congolese', 'Conservative', 'Costa Rican', 'Croat', 'Cuban', 'Cypriot',
         'Czech', 'Dane', 'Danish',
         'Democrat', 'Democratic', 'Djibouti', 'Dominican', 'Dutch', 'East European', 'Ecuadorean', 'Egyptian',
         'Emirati', 'English', 'Equatoguinean',
         'Equatorial Guinean', 'Eritrean', 'Estonian', 'Ethiopian', 'Eurasian', 'European', 'Fijian', 'Filipino',
         'Finn', 'Finnish', 'French',
         'Gabonese', 'Gambian', 'Georgian', 'German', 'Germanic', 'Ghanaian', 'Greek', 'Greenlander', 'Grenadan',
         'Grenadian', 'Guadeloupean', 'Guatemalan',
         'Guinea-Bissauan', 'Guinean', 'Guyanese', 'Haitian', 'Hawaiian', 'Hindu', 'Hinduist', 'Hispanic', 'Honduran',
         'Hungarian', 'Icelander', 'Indian',
         'Indonesian', 'Iranian', 'Iraqi', 'Irish', 'Islamic', 'Islamist', 'Israeli', 'Israelite', 'Italian', 'Ivorian',
         'Jain', 'Jamaican', 'Japanese',
         'Jew', 'Jewish', 'Jordanian', 'Kazakhstani', 'Kenyan', 'Kirghiz', 'Korean', 'Kurd', 'Kurdish', 'Kuwaiti',
         'Kyrgyz', 'Labour', 'Latin',
         'Latin American', 'Latvian', 'Lebanese', 'Liberal', 'Liberian', 'Libyan', 'Liechtensteiner', 'Lithuanian',
         'Londoner', 'Luxembourger',
         'Macedonian', 'Malagasy', 'Malawian', 'Malaysian', 'Maldivan', 'Malian', 'Maltese', 'Manxman', 'Marshallese',
         'Martinican', 'Martiniquais',
         'Marxist', 'Mauritanian', 'Mauritian', 'Mexican', 'Micronesian', 'Moldovan', 'Mongolian', 'Montenegrin',
         'Montserratian', 'Moroccan',
         'Motswana', 'Mozambican', 'Muslim', 'Myanmarese', 'Namibian', 'Nationalist', 'Nazi', 'Nauruan', 'Nepalese',
         'Netherlander', 'New Yorker',
         'New Zealander', 'Nicaraguan', 'Nigerian', 'Nordic', 'North American', 'North Korean', 'Norwegian', 'Orthodox',
         'Pakistani', 'Palauan',
         'Palestinian', 'Panamanian', 'Papua New Guinean', 'Paraguayan', 'Parisian', 'Peruvian', 'Philistine', 'Pole',
         'Polish', 'Portuguese',
         'Protestant', 'Puerto Rican', 'Qatari', 'Republican', 'Roman', 'Romanian', 'Russian', 'Rwandan',
         'Saint Helenian', 'Saint Lucian',
         'Saint Vincentian', 'Salvadoran', 'Sammarinese', 'Samoan', 'San Marinese', 'Sao Tomean', 'Saudi',
         'Saudi Arabian', 'Scandinavian', 'Scottish',
         'Senegalese', 'Serb', 'Serbian', 'Shia', 'Shiite', 'Sierra Leonean', 'Sikh', 'Singaporean', 'Slovak',
         'Slovene', 'Social-Democrat', 'Socialist',
         'Somali', 'South African', 'South American', 'South Korean', 'Soviet', 'Spaniard', 'Spanish', 'Sri Lankan',
         'Sudanese', 'Sunni',
         'Surinamer', 'Swazi', 'Swede', 'Swedish', 'Swiss', 'Syrian', 'Taiwanese', 'Tajik', 'Tanzanian', 'Taoist',
         'Texan', 'Thai', 'Tibetan',
         'Tobagonian', 'Togolese', 'Tongan', 'Tunisian', 'Turk', 'Turkish', 'Turkmen(s)', 'Tuvaluan', 'Ugandan',
         'Ukrainian', 'Uruguayan', 'Uzbek',
         'Uzbekistani', 'Venezuelan', 'Vietnamese', 'Vincentian', 'Virgin Islander', 'Welsh', 'West European',
         'Western', 'Yemeni', 'Yemenite',
         'Yugoslav', 'Zambian', 'Zimbabwean', 'Zionist'}

# Facilities
FACILITIES = {"Palace", "Temple", "Gate", "Museum", "Bridge", "Road", "Airport", "Hospital", "School", "Tower",
              "Station", "Avenue",
              "Prison", "Building", "Plant", "Shopping Center", "Shopping Centre", "Mall", "Church", "Synagogue",
              "Mosque", "Harbor", "Harbour",
              "Rail", "Railway", "Metro", "Tram", "Highway", "Tunnel", 'House', 'Field', 'Hall', 'Place', 'Freeway',
              'Wall', 'Square', 'Park',
              'Hotel'}

# Legal documents
LEGAL = {"Law", "Agreement", "Act", 'Bill', "Constitution", "Directive", "Treaty", "Code", "Reform", "Convention",
         "Resolution", "Regulation",
         "Amendment", "Customs", "Protocol", "Charter"}

# event names
EVENTS = {"War", "Festival", "Show", "Massacre", "Battle", "Revolution", "Olympics", "Games", "Cup", "Week", "Day",
          "Year", "Series"}

# Names of languages
LANGUAGES = {'Afar', 'Abkhazian', 'Avestan', 'Afrikaans', 'Akan', 'Amharic', 'Aragonese', 'Arabic', 'Aramaic',
             'Assamese', 'Avaric', 'Aymara',
             'Azerbaijani', 'Bashkir', 'Belarusian', 'Bulgarian', 'Bambara', 'Bislama', 'Bengali', 'Tibetan', 'Breton',
             'Bosnian', 'Cantonese',
             'Catalan', 'Chechen', 'Chamorro', 'Corsican', 'Cree', 'Czech', 'Chuvash', 'Welsh', 'Danish', 'German',
             'Divehi', 'Dzongkha', 'Ewe',
             'Greek', 'English', 'Esperanto', 'Spanish', 'Castilian', 'Estonian', 'Basque', 'Persian', 'Fulah',
             'Filipino', 'Finnish', 'Fijian', 'Faroese',
             'French', 'Western Frisian', 'Irish', 'Gaelic', 'Galician', 'Guarani', 'Gujarati', 'Manx', 'Hausa',
             'Hebrew', 'Hindi', 'Hiri Motu',
             'Croatian', 'Haitian', 'Hungarian', 'Armenian', 'Herero', 'Indonesian', 'Igbo', 'Inupiaq', 'Ido',
             'Icelandic', 'Italian', 'Inuktitut',
             'Japanese', 'Javanese', 'Georgian', 'Kongo', 'Kikuyu', 'Kuanyama', 'Kazakh', 'Kalaallisut', 'Greenlandic',
             'Central Khmer', 'Kannada',
             'Korean', 'Kanuri', 'Kashmiri', 'Kurdish', 'Komi', 'Cornish', 'Kirghiz', 'Latin', 'Luxembourgish', 'Ganda',
             'Limburgish', 'Lingala', 'Lao',
             'Lithuanian', 'Luba-Katanga', 'Latvian', 'Malagasy', 'Marshallese', 'Maori', 'Macedonian', 'Malayalam',
             'Mongolian', 'Marathi', 'Malay',
             'Maltese', 'Burmese', 'Nauru', 'Bokmål', 'Norwegian', 'Ndebele', 'Nepali', 'Ndonga', 'Dutch', 'Flemish',
             'Nynorsk', 'Navajo', 'Chichewa',
             'Occitan', 'Ojibwa', 'Oromo', 'Oriya', 'Ossetian', 'Punjabi', 'Pali', 'Polish', 'Pashto', 'Portuguese',
             'Quechua', 'Romansh', 'Rundi',
             'Romanian', 'Russian', 'Kinyarwanda', 'Sanskrit', 'Sardinian', 'Sindhi', 'Sami', 'Sango', 'Sinhalese',
             'Slovak', 'Slovenian', 'Samoan',
             'Shona', 'Somali', 'Albanian', 'Serbian', 'Swati', 'Sotho', 'Sundanese', 'Swedish', 'Swahili', 'Tamil',
             'Telugu', 'Tajik', 'Thai',
             'Tigrinya', 'Turkmen', 'Taiwanese', 'Tagalog', 'Tswana', 'Tonga', 'Turkish', 'Tsonga', 'Tatar', 'Twi',
             'Tahitian', 'Uighur', 'Ukrainian',
             'Urdu', 'Uzbek', 'Venda', 'Vietnamese', 'Volapük', 'Walloon', 'Wolof', 'Xhosa', 'Yiddish', 'Yoruba',
             'Zhuang', 'Mandarin',
             'Mandarin Chinese', 'Chinese', 'Zulu'}

LEGAL_SUFFIXES = {
    'ltd',  # Limited ~13.000
    'llc',  # limited liability company (UK)
    'ltda',  # limitada (Brazil, Portugal)
    'inc',  # Incorporated ~9700
    'co ltd',  # Company Limited ~9200
    'corp',  # Corporation ~5200
    'sa',  # Spółka Akcyjna (Poland), Société Anonyme (France)  ~3200
    'plc',  # Public Limited Company (Great Britain) ~2100
    'ag',  # Aktiengesellschaft (Germany) ~1000
    'gmbh',  # Gesellschaft mit beschränkter Haftung  (Germany)
    'bhd',  # Berhad (Malaysia) ~900
    'jsc',  # Joint Stock Company (Russia) ~900
    'co',  # Corporation/Company ~900
    'ab',  # Aktiebolag (Sweden) ~800
    'ad',  # Akcionarsko Društvo (Serbia), Aktsionerno Drujestvo (Bulgaria) ~600
    'tbk',  # Terbuka (Indonesia) ~500
    'as',  # Anonim Şirket (Turkey), Aksjeselskap (Norway) ~500
    'pjsc',  # Public Joint Stock Company (Russia, Ukraine) ~400
    'spa',  # Società Per Azioni (Italy) ~300
    'nv',  # Naamloze vennootschap (Netherlands, Belgium) ~230
    'dd',  # Dioničko Društvo (Croatia) ~220
    'a s',  # a/s (Denmark), a.s (Slovakia) ~210
    'oao',  # Открытое акционерное общество (Russia) ~190
    'asa',  # Allmennaksjeselskap (Norway) ~160
    'ojsc',  # Open Joint Stock Company (Russia) ~160
    'lp',  # Limited Partnership (US) ~140
    'llp',  # limited liability partnership
    'oyj',  # julkinen osakeyhtiö (Finland) ~120
    'de cv',  # Capital Variable (Mexico) ~120
    'se',  # Societas Europaea (Germany) ~100
    'kk',  # kabushiki gaisha (Japan)
    'aps',  # Anpartsselskab (Denmark)
    'cv',  # commanditaire vennootschap (Netherlands)
    'sas',  # société par actions simplifiée (France)
    'sro',  # Spoločnosť s ručením obmedzeným (Slovakia)
    'oy',  # Osakeyhtiö (Finland)
    'kg',  # Kommanditgesellschaft (Germany)
    'bv',  # Besloten Vennootschap (Netherlands)
    'sarl',  # société à responsabilité limitée (France)
    'srl',  # Società a responsabilità limitata (Italy)
    'sl'  # Sociedad Limitada (Spain)
}
# Generic words that may appear in official company names but are sometimes skipped when mentioned in news articles (e.g. Nordea Bank -> Nordea)
GENERIC_TOKENS = {"International", "Group", "Solutions", "Technologies", "Management", "Association", "Associates",
                  "Partners",
                  "Systems", "Holdings", "Services", "Bank", "Fund", "Stiftung", "Company"}

# List of tokens that are typically lowercase even when they occur in capitalised segments (e.g. International Council of Shopping Centers)
LOWERCASED_TOKENS = {"'s", "-", "a", "an", "the", "at", "by", "for", "in", "of", "on", "to", "up", "and"}

# Prefixes to family names that are often in lowercase
NAME_PREFIXES = {"-", "von", "van", "de", "di", "le", "la", "het", "'t'", "dem", "der", "den", "d'", "ter"}
